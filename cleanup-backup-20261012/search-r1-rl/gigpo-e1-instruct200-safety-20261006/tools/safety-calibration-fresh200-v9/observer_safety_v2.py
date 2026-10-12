"""Passive observation isolation; original computation errors are never caught here."""
from contextlib import contextmanager


def note(record, label, error):
    # Error formatting/journaling itself is passive and must not skip a callable.
    try:
        try:
            description = repr(error)
        except BaseException:
            description = type(error).__name__ + ' (error formatting failed)'
        errors = record.get('observer_errors')
        if not isinstance(errors, list):
            errors = ['Observer error journal was malformed']
            record['observer_errors'] = errors
        errors.append(label + ': ' + description)
    except BaseException:
        try:
            record['_observer_recording_failed'] = True
        except BaseException:
            pass  # Missing required capture is separately refused at the matched boundary.



def attempt(record, label, operation):
    try:
        return operation()
    except BaseException as error:
        note(record, label, error)
        return None


@contextmanager
def patches(record):
    """Every attempted mutation has a prior rollback entry, even partial setattr."""
    installed = []

    def install(target, name, value, *, mapping=False):
        try:
            existed = name in target if mapping else name in vars(target)
            original = target.get(name) if mapping else getattr(target, name)
            installed.append((target, name, existed, original, mapping))
            if mapping:
                target[name] = value
            else:
                setattr(target, name, value)
            return True
        except BaseException as error:
            note(record, 'install/' + name, error)
            return False

    try:
        yield install
    finally:
        for target, name, existed, original, mapping in reversed(installed):
            def restore():
                if mapping:
                    if existed: target[name] = original
                    else: target.pop(name, None)
                elif existed:
                    setattr(target, name, original)
                elif name in vars(target):
                    delattr(target, name)
            attempt(record, 'restore/' + name, restore)


def observer_failures(calls):
    errors = []
    for call in calls:
        if call.get('_observer_recording_failed'):
            errors.append('Observer error journal itself failed')
        errors.extend(call.get('observer_errors', []))
        for key in ('actual_normalizer_calls', 'actual_sampler_calls'):
            for row in call.get(key, []):
                if row.get('_observer_recording_failed'):
                    errors.append('Normalizer error journal itself failed')
                errors.extend(row.get('observer_errors', []))
    return errors


def install_pre_hook(record, module, callback, label):
    """Restore partially installed PyTorch hook registries even without a handle."""
    snapshots, handle = {}, None
    try:
        for key in ('_forward_pre_hooks', '_forward_pre_hooks_with_kwargs'):
            snapshots[key] = dict(getattr(module, key))
        handle = module.register_forward_pre_hook(callback, with_kwargs=True)
    except BaseException as error:
        note(record, label + '/install', error)

    def cleanup():
        if handle is not None:
            attempt(record, label + '/handle_remove', handle.remove)
        for key, before in snapshots.items():
            def restore_registry():
                registry = getattr(module, key)
                # Only this serial observation may have added hooks. Preserve the
                # original entries themselves, including callback object identity.
                for identifier in set(registry) - set(before):
                    del registry[identifier]
                for identifier, callback_value in before.items():
                    registry[identifier] = callback_value
                if set(registry) != set(before) or any(registry[k] is not v for k, v in before.items()):
                    raise ValueError('Original hook registry identity was not restored')
            attempt(record, label + '/partial_registry_restore/' + key, restore_registry)
    return cleanup
