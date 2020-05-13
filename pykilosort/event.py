# -*- coding: utf-8 -*-

"""Simple event system."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from contextlib import contextmanager
import logging
import re
from functools import partial

logger = logging.getLogger(__name__)


#------------------------------------------------------------------------------
# Event system
#------------------------------------------------------------------------------

class EventEmitter(object):
    """Singleton class that emits events and accepts registered callbacks.

    Example
    -------

    ```python
    class MyClass(EventEmitter):
        def f(self):
            self.emit('my_event', 1, key=2)

    o = MyClass()

    # The following function will be called when `o.f()` is called.
    @o.connect
    def on_my_event(arg, key=None):
        print(arg, key)

    ```

    """

    def __init__(self):
        self.reset()
        self.is_silent = False

    def set_silent(self, silent):
        """Set whether to silence the events."""
        self.is_silent = silent

    def reset(self):
        """Remove all registered callbacks."""
        self._callbacks = []

    def _get_on_name(self, func):
        """Return `eventname` when the function name is `on_<eventname>()`."""
        r = re.match("^on_(.+)$", func.__name__)
        if r:
            event = r.group(1)
        else:
            raise ValueError("The function name should be "
                             "`on_<eventname>`().")
        return event

    @contextmanager
    def silent(self):
        """Prevent all callbacks to be called if events are raised
        in the context manager.
        """
        self.is_silent = not(self.is_silent)
        yield
        self.is_silent = not(self.is_silent)

    def connect(self, func=None, event=None, **kwargs):
        """Register a callback function to a given event.

        To register a callback function to the `spam` event, where `obj` is
        an instance of a class deriving from `EventEmitter`:

        ```python
        @obj.connect
        def on_spam(arg1, arg2):
            pass
        ```

        This is called when `obj.emit('spam', arg1, arg2)` is called.

        Several callback functions can be registered for a given event.

        The registration order is conserved and may matter in applications.

        """
        if func is None:
            return partial(self.connect, event=event, **kwargs)

        # Get the event name from the function.
        if event is None:
            event = self._get_on_name(func)

        # We register the callback function.
        self._callbacks.append((event, func, kwargs))

        return func

    def unconnect(self, *items):
        """Unconnect specified callback functions."""
        self._callbacks = [
            (event, f, kwargs)
            for (event, f, kwargs) in self._callbacks
            if f not in items]

    def emit(self, event, *args, **kwargs):
        """Call all callback functions registered with an event.

        Any positional and keyword arguments can be passed here, and they will
        be forwarded to the callback functions.

        Return the list of callback return results.

        """
        if self.is_silent:
            return
        logger.log(
            5, "Emit %s(%s, %s)", event,
            ', '.join(map(str, args)), ', '.join('%s=%s' % (k, v) for k, v in kwargs.items()))
        # Call the last callback if this is a single event.
        single = kwargs.pop('single', None)
        res = []
        # Put `last=True` callbacks at the end.
        callbacks = [c for c in self._callbacks if not c[-1].get('last', None)]
        callbacks += [c for c in self._callbacks if c[-1].get('last', None)]
        for e, f, k in callbacks:
            if e == event:
                f_name = getattr(f, '__qualname__', getattr(f, '__name__', str(f)))
                logger.log(5, "Callback %s.", f_name)
                res.append(f(*args, **kwargs))
                if single:
                    return res[-1]
        return res


#------------------------------------------------------------------------------
# Global event system
#------------------------------------------------------------------------------

_EVENT = EventEmitter()

emit = _EVENT.emit
connect = _EVENT.connect
unconnect = _EVENT.unconnect
silent = _EVENT.silent
set_silent = _EVENT.set_silent
reset = _EVENT.reset
