"""Public adjudication metadata for the current 1729 rally, never hidden cards."""


def note_revolution(room, cards):
    room.revolution_serial = getattr(room, 'revolution_serial', 0) + 1
    room.revolution_recovery_context = (
        dict(serial=room.revolution_serial, active=True,
             trigger_ids=tuple(str(c['card_id']) for c in cards))
        if room.reverse_order else None)


def note_field_flow(room):
    context = getattr(room, 'revolution_recovery_context', None)
    room.revolution_recovery_context = (
        dict(context, active=False, closing_player=getattr(room, 'last_play_player_id', None),
             closing_ids=tuple(str(c['card_id']) for c in room.field))
        if context and context['active'] else None)
