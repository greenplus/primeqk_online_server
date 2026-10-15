"""Knowledge-relative AND/OR certificates for a public, continuing rally.

Only visible cards in the current rally are locked. Materials return immediately;
every field flow clears locks and aggregate bounds. Replies are grouped by the
next planned value and physical face consumption. Each common continuation is
proved against a conservative state containing every member of that group.
"""
from dataclasses import dataclass, field as dataclass_field
from collections import defaultdict
from itertools import chain
from functools import lru_cache
import time

from .opening_master import (CAPACITY, ZERO, Move, Plan, Template, counts,
                             subtract, hand_counts, _realizations, bind_move)
from .post_all_out_master import Solver, Config, SearchLimit, catalog, knowledge_key, move_resource_key
import cpu_player as diamond
from registered_primes import registered_value_encodings


# Bounded pure tuple operations, shared by equivalent rally states. No proof
# or timeout result is memoized here, and no CPU/room is retained by the cache.
subtract = lru_cache(maxsize=32768)(subtract)


@lru_cache(maxsize=32768)
def add(a, b):
    return tuple(x+y for x,y in zip(a,b))


def faces(hand):
    return hand[0] + sum(hand[10:14])


def visible_need(move):
    return counts(move.physical[:move.size])


@dataclass
class Reply:
    move: Move
    visible: tuple
    opponent_counts: tuple
    variants: int = 1
    consumption_variants: set = dataclass_field(default_factory=set)

    def __post_init__(self):
        self.consumption_variants.add((faces(self.move.need), self.move.need[13], self.move.need[0]))

    @property
    def key(self):
        return (self.move.value, self.visible, self.opponent_counts)

    def matches(self, value, visible, opponent_count):
        return value == self.move.value and visible == self.visible and opponent_count in self.opponent_counts

    def summary(self):
        physical = self.move.need
        material = subtract(physical, self.visible)
        return dict(response=self.move.summary(), visible_counts=list(self.visible),
                    face_consumption=faces(physical), k_consumption=physical[13], x_consumption=physical[0],
                    locked_faces=faces(self.visible), locked_k=self.visible[13], locked_x=self.visible[0],
                    returned_material_faces=faces(material), opponent_counts=list(self.opponent_counts),
                    equivalent_physical_variants=self.variants,
                    all_consumption_face_k_x=[list(v) for v in sorted(self.consumption_variants)])


@dataclass
class ReplyGroup:
    threshold: int
    band: str
    face_count: int
    representative: Reply
    min_value: int
    common_visible: tuple
    opponent_counts: tuple
    remaining_limits: tuple
    min_kx: int
    members: int
    finish_witness: object = None
    special: bool = False

    @property
    def move(self):
        return self.representative.move

    @property
    def key(self):
        return self.band, self.face_count

    def matches(self, value, visible, opponent_count):
        return (self.min_value <= value <= self.move.value
                and ('below-M' if value < self.threshold else 'at-or-above-M') == self.band
                and faces(visible) == self.face_count
                and visible[0]+visible[13] >= self.min_kx
                and subtract(visible,self.common_visible) is not None
                and opponent_count in self.opponent_counts)

    def summary(self):
        return dict(class_key=list(self.key),threshold=self.threshold,band=self.band,face_count=self.face_count,
                    min_value=str(self.min_value),max_value=str(self.move.value),
                    representative=self.representative.summary(),common_visible=list(self.common_visible),
                    opponent_counts=list(self.opponent_counts),remaining_limits=list(self.remaining_limits),
                    min_kx_consumption=self.min_kx,member_classes=self.members,
                    finish_witness=self.finish_witness,special=self.special)


@dataclass
class Node:
    id: int
    move: Move
    held: tuple
    locked: tuple
    opponent_counts: tuple
    field: tuple | None
    pass_node: object = None
    replies: tuple = ()  # (Reply, Node)
    terminal: bool = False
    cut: bool = False
    limits: tuple | None = None


@dataclass
class DoublePlan:
    steps: tuple
    root: Node
    candidate: dict
    model: str = 'double_ceiling_mate'
    pass_sequence: tuple = ()

    def summary(self):
        base = Plan('certified_win', self.steps, 0, self.model, 0).summary()
        return base | dict(double_ceiling_mate=self.model=='double_ceiling_mate', certificate_root=self.root.id,
                           candidate_id=self.candidate['id'])


class DoubleCeilingSolver(Solver):
    def __init__(self, cpu, room, config=Config(), prepared=None):
        super().__init__(cpu, room, Config(hand_sizes=config.hand_sizes,
                         max_nodes=config.double_max_nodes, budget_ms=config.budget_ms))
        self.reuse_prepared(prepared)
        self.response_templates = defaultdict(list)
        templates = list(catalog(*self.knowledge, for_responses=True))
        templates += [Template('prime',1729,r) for r in registered_value_encodings(1729,max_cards=4)]
        for t in templates:
            self.response_templates[len(t.visible)].append(t)
        self.reply_cache = {}
        self.group_cache = {}
        self.respond_cache = {}
        self.ordered_fit_cache = {}
        self.free_cache = {}
        self.candidates = []
        self.certificate_nodes = {}
        self.next_node = 0
        self.best = None
        self.completed_sizes = []
        self.active_candidate = None
        self.swap_keys = set()
        self.realization_cache = lru_cache(maxsize=4096)(
            lambda template, available: tuple(_realizations(template, available)))

    def new_node(self, move, held, locked, opponent_counts, field, **kw):
        self.next_node += 1
        node = Node(self.next_node, move, held, locked, opponent_counts, field, **kw)
        self.certificate_nodes[node.id] = node
        return node

    def free_certificate(self, held):
        # A field flow returns BOTH players' entire reserve. Never carry a
        # resource discount into this existing Certified Win search.
        if held not in self.free_cache:
            sequence = self.control_tail(held)
            if sequence is None:
                self.free_cache[held] = None
            else:
                current = held
                records = []
                for move in sequence:
                    self.tick()
                    rest = subtract(current, move.need)
                    records.append((move,current,rest))
                    current = rest
                child = None
                for move,current,rest in reversed(records):
                    child = self.new_node(move,current,ZERO,(),None,pass_node=child,
                                          terminal=rest == ZERO,cut=move.cut)
                self.free_cache[held] = child
        return self.free_cache[held]

    def pool_after_move(self, move, held, locked, limits):
        rest = subtract(held,move.need)
        blocked = add(locked,visible_need(move))
        available = subtract(subtract(CAPACITY,rest),blocked)
        material = subtract(move.need,visible_need(move))
        caps = (faces(available),available[0]+available[13])
        if limits is not None:
            caps = (min(caps[0],limits[0]+faces(material)),
                    min(caps[1],limits[1]+material[0]+material[13]))
        return available,caps

    def reply_classes(self, move, held, locked, opponent_counts, limits=None):
        available,caps = self.pool_after_move(move,held,locked,limits)
        key = (move.size,move.value,available,opponent_counts,caps)
        if key not in self.reply_cache:
            grouped = {}
            # Permit one draw even if the deck might be empty (safe superset).
            before_counts = set(chain.from_iterable((n,n+1) for n in opponent_counts))
            for template in self.response_templates[move.size]:
                self.tick()
                if template.kind != 'cut' and template.value <= move.value:
                    continue
                for reply in self.realization_cache(template,available):
                    self.tick()
                    if faces(reply.need)>caps[0] or reply.need[0]+reply.need[13]>caps[1]:
                        continue
                    after = tuple(sorted(n-reply.used_count for n in before_counts if n >= reply.used_count))
                    if not after:
                        continue
                    item = Reply(reply,visible_need(reply),after)
                    if item.key in grouped:
                        grouped[item.key].variants += 1
                        grouped[item.key].consumption_variants.update(item.consumption_variants)
                    else:
                        grouped[item.key] = item
            # Strong values and low resource costs first; no value-band pruning.
            self.reply_cache[key] = tuple(sorted(grouped.values(),key=lambda r:(
                -(r.move.value or 10**100),faces(r.visible),r.key)))
        return self.reply_cache[key]

    def response_groups(self, move, held, locked, opponent_counts, threshold, limits=None):
        key=(move_resource_key(move),held,locked,opponent_counts,threshold,limits)
        if key not in self.group_cache:
            _,caps=self.pool_after_move(move,held,locked,limits)
            grouped=defaultdict(list)
            for reply in self.reply_classes(move,held,locked,opponent_counts,limits):
                grouped['below-M' if reply.move.value<threshold else 'at-or-above-M',faces(reply.visible)].append(reply)
            result=[]
            for (band,r),members in grouped.items():
                # Componentwise common locks + global face/KX caps contain every
                # member's future pool. Max R and union of hand counts may be
                # mutually unattainable; proving that larger state is safe.
                common=tuple(min(m.visible[i] for m in members) for i in range(14))
                min_kx=min(m.visible[0]+m.visible[13] for m in members)
                representative=max(members,key=lambda m:m.move.value)
                winning=next((m for m in members if 0 in m.opponent_counts),None)
                result.append(ReplyGroup(threshold,band,r,representative,min(m.move.value for m in members),common,
                    tuple(sorted({n for m in members for n in m.opponent_counts})),
                    (caps[0]-r,caps[1]-min_kx),min_kx,len(members),
                    winning.summary() if winning else None,
                    any(m.move.cut or m.move.value==1729 for m in members)))
            self.group_cache[key]=tuple(sorted(result,key=lambda g:(g.band=='below-M',g.face_count)))
        return self.group_cache[key]

    def prove_move(self, move, held, locked, opponent_counts, field, preferred, mode, audit=None, limits=None):
        self.tick()
        rest = subtract(held,move.need)
        if rest is None or (field and (move.size != field[0] or
                                      move.template.kind != 'cut' and move.value <= field[1])):
            return None, dict(status='unproved',reason='illegal-continuation')
        if rest == ZERO:
            return self.new_node(move,held,locked,opponent_counts,field,terminal=True,limits=limits), None
        passed = self.free_certificate(rest)
        if passed is None:
            return None, dict(status='unproved',reason='pass-recovery-remainder-not-certified')
        if move.cut:
            if audit is not None:
                audit.update(response_class_count=0,responses=[])
            return self.new_node(move,held,locked,opponent_counts,field,pass_node=passed,cut=True,limits=limits), None
        if self.certain(move,held,locked=locked,limits=limits):
            if audit is not None:
                audit.update(response_class_count=0,responses=[])
            return self.new_node(move,held,locked,opponent_counts,field,pass_node=passed,limits=limits), None
        threshold=preferred[0].value if preferred else 10**100
        replies = self.response_groups(move,held,locked,opponent_counts,threshold,limits)
        if not replies:
            return None,dict(status='unproved',reason='no-policy-certain-control')
        next_locked = add(locked,visible_need(move))
        records = [r.summary() | dict(status='not-visited') for r in replies]
        if audit is not None:
            audit.update(response_class_count=len(replies),responses=records)
        children = []
        for reply,record in zip(replies,records):
            self.tick()
            reduced = add(next_locked,reply.common_visible)
            record.update(opponent_max_faces=reply.remaining_limits[0],opponent_max_kx=reply.remaining_limits[1])
            if 0 in reply.opponent_counts:
                failure = dict(status='refuted',reason='opponent-immediate-finish',response=reply.summary())
                record.update(status='refuted',failure=failure)
                return None,failure
            if reply.special:
                failure = dict(status='unproved',reason='opponent-cut-or-revolution',response=reply.summary())
                record.update(status='unproved',failure=failure)
                return None,failure
            child,failure = self.respond(rest,reduced,reply.opponent_counts,
                                        (reply.move.size,reply.move.value),preferred,mode,limits=reply.remaining_limits)
            if child is None:
                record.update(status='unproved',failure=failure)
                return None,dict(status='unproved',reason='unresolved-response',response=reply.summary(),detail=failure)
            record.update(status='certified',continuation_root=child.id,
                          alternative_first=child.move.summary())
            children.append((reply,child))
        return self.new_node(move,held,locked,opponent_counts,field,
                             pass_node=passed,replies=tuple(children),limits=limits),None

    def respond(self, held, locked, opponent_counts, field, preferred, mode, limits=None):
        key = (held,locked,opponent_counts,field,tuple(m.key for m in preferred),mode,tuple(sorted(self.swap_keys)),limits)
        if key in self.respond_cache:
            return self.respond_cache[key]
        self.tick()
        moves = []
        if preferred:
            moves.append(preferred[0])
        if mode in ('swap','repartition'):
            moves.extend(m for m in preferred[1:] if m.key in self.swap_keys)
        if mode == 'repartition':
            # Same displayed count; the face demand emerges from the exact
            # remaining public resource pool in certain(), never a 5+5 rule.
            fit_key = (held, field[0])
            if fit_key not in self.ordered_fit_cache:
                self.ordered_fit_cache[fit_key] = tuple(sorted(self.fitting_moves(held,field[0]),
                    key=lambda m:(-m.value,-faces(m.need),m.key)))
            moves.extend(self.ordered_fit_cache[fit_key])
        seen = set()
        failure = dict(status='unproved',reason='no-certified-legal-continuation')
        for move in moves:
            self.tick()
            if move.key in seen:
                continue
            seen.add(move.key)
            if (move.template.kind != 'cut' and move.value <= field[1]
                    or subtract(held,move.need) is None):
                continue
            suffix = tuple(m for m in preferred if m.key != move.key)
            # Once a repartition consumes other planned cards, the old tuple
            # is only an ordering hint. Every move is checked against held.
            result,why = self.prove_move(move,held,locked,opponent_counts,field,suffix,mode,limits=limits)
            if result is not None:
                self.respond_cache[key] = (result,None)
                return result,None
            failure = why
        self.respond_cache[key] = (None,failure)
        return None,failure

    def nominal_candidates(self, hand, size, *, inclusive=True):
        trumps = sorted(self.by_size[size],key=lambda m:(-(m.value or 10**100),m.key))
        for trump in trumps:
            self.tick()
            reason = self.ceiling_reason(trump,hand)
            if reason is None or not self.certain(trump,hand,inclusive=inclusive):
                continue
            after_t = subtract(hand,trump.need)
            if sum(after_t) < 2*size:
                continue
            # The same T value may use different X/material resources.
            for middle in sorted(self.fitting_moves(after_t,size),key=lambda m:(-m.value,m.key)):
                self.tick()
                if middle.value >= (trump.value or 10**100) or middle.cut:
                    continue
                left = subtract(after_t,middle.need)
                if left is None or sum(left) < size:
                    continue
                for lead in self.fitting_moves(left,size):
                    self.tick()
                    if lead.cut or lead.value >= middle.value:
                        continue
                    tail_hand = subtract(left,lead.need)
                    if tail_hand is None:
                        continue
                    tail = self.control_tail(tail_hand)
                    if tail is None:
                        self.candidates.append(dict(id=len(self.candidates)+1,hand_size=sum(hand),face_count=faces(hand),n=size,
                            L=lead.summary(),M=middle.summary(),T=trump.summary(),T_certain_reason=reason,
                            remainder=list(tail_hand),remainder_steps=None,status='unproved',mate=False,timeout=False,
                            attempts=[],unresolved_response=dict(status='unproved',reason='remainder-not-certified')))
                        continue
                    yield lead,middle,trump,tail,tail_hand,reason

    def evaluate_double(self, hand):
        self.original_hand = hand
        if (sum(hand) not in self.config.hand_sizes or faces(hand) < 10
                or getattr(self.room,'reverse_order',False) or getattr(self.room,'field',[])
                or self.opponent_count is None):
            return None
        try:
            if not self.moves:
                self.prepare(hand)
            for size in (6,7):
                for lead,middle,trump,tail,tail_hand,reason in self.nominal_candidates(hand,size):
                    candidate = dict(id=len(self.candidates)+1,hand_size=sum(hand),face_count=faces(hand),n=size,
                                     L=lead.summary(),M=middle.summary(),T=trump.summary(),
                                     T_certain_reason=reason,remainder=list(tail_hand),
                                     remainder_steps=[m.summary() for m in tail],
                                     status='unproved',mate=False,timeout=False,attempts=[])
                    self.candidates.append(candidate)
                    self.active_candidate = candidate
                    self.swap_keys = {middle.key,trump.key}
                    modes = ('fixed','swap','repartition') if faces(hand)>=11 else ('fixed','repartition')
                    for mode in modes:
                        attempt = dict(mode=mode,response_class_count=None,responses=[])
                        candidate['attempts'].append(attempt)
                        root,failure = self.prove_move(lead,hand,ZERO,(self.opponent_count,),None,
                                                      (middle,trump)+tail,mode,attempt)
                        if root is not None:
                            model = 'double_ceiling_mate' if root.replies else 'public_control_sequence'
                            candidate.update(status='certified',mate=True,double_ceiling_mate=bool(root.replies),
                                             certificate_root=root.id,mode=mode)
                            self.best = DoublePlan((lead,middle,trump)+tail,root,candidate,model=model)
                            return self.best
                        attempt.update(failure=failure)
                        candidate['unresolved_response'] = failure
                    # Refutation applies to this candidate only when a reply
                    # wins immediately, independent of all our continuation choices.
                    if candidate['unresolved_response'].get('reason') == 'opponent-immediate-finish':
                        candidate['status'] = 'refuted'
                    self.active_candidate = None
                self.completed_sizes.append(size)
        except SearchLimit:
            self.timed_out = True
            if self.active_candidate is not None:
                self.active_candidate.update(timeout=True,status='unproved')
        return None

    def certificate(self, root):
        nodes = {}
        def visit(node):
            if node is None or str(node.id) in nodes:
                return
            nodes[str(node.id)] = dict(move=node.move.summary(),held=list(node.held),locked=list(node.locked),
                opponent_counts=list(node.opponent_counts),field=list(node.field) if node.field else None,
                terminal=node.terminal,cut=node.cut,pass_node=node.pass_node.id if node.pass_node else None,
                limits=list(node.limits) if node.limits is not None else None,
                replies=[r.summary() | dict(child=n.id) for r,n in node.replies])
            visit(node.pass_node)
            for _,child in node.replies:
                visit(child)
        visit(root)
        return dict(root=root.id,nodes=nodes)

    def summary_double(self):
        return dict(model='double_ceiling_mate',proof_scope='registered-knowledge',
                    elapsed_ms=1000*(time.perf_counter()-self.started),nodes=self.nodes,
                    timed_out=self.timed_out,completed_sizes=self.completed_sizes,
                    hand_size=sum(self.original_hand),face_count=faces(self.original_hand),
                    candidates=self.candidates,best=self.best.summary() if self.best else None,
                    certificate=self.certificate(self.best.root) if self.best else None,
                    response_compression='below-M-or-at-least-M/by-visible-physical-face-count',
                    certain_policy='listed-or-leading-K-lock-or-face-count-lock')


def commit_node(cpu,room,active,node):
    candidate = bind_move(node.move,cpu.hand)
    active.update(node=node,expected=subtract(hand_counts(cpu.hand),node.move.need))
    cpu.post_all_out_master_active = None if node.terminal else active
    cpu.diamond_last_route_kind = 'post-all-out-master-'+active['model']
    cpu.diamond_last_certainty = 'certain'
    cpu.diamond_last_return_probability = 0.0
    diamond.diamond_update_opening_position_history(cpu,room)
    diamond.diamond_remember_cards(cpu,diamond.candidate_consumed_cards(candidate))
    return diamond.platinum_commit_play(cpu,diamond.candidate_to_action(candidate))


def continue_double(cpu,room,active):
    previous = active['node']
    if not room.field:
        node = previous.pass_node
    else:
        physical = hand_counts(room.field)
        value = int(room.last_number)
        opponent_count = diamond.platinum_opponent_hand_count(cpu,room)
        matches = [child for reply,child in previous.replies if reply.matches(value,physical,opponent_count)]
        node = matches[0] if matches else None
    if node is None:
        cpu.post_all_out_master_active = None
        return None
    cpu.post_all_out_master_trace = dict(continuation=True,model=active['model'],certificate_node=node.id)
    return commit_node(cpu,room,active,node)
