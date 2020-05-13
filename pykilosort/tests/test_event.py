# -*- coding: utf-8 -*-

"""Test event system."""

#------------------------------------------------------------------------------
# Imports
#------------------------------------------------------------------------------

from pytest import raises

from ..event import EventEmitter


#------------------------------------------------------------------------------
# Test event system
#------------------------------------------------------------------------------

def test_event_system():
    ev = EventEmitter()

    _list = []

    with raises(ValueError):
        ev.connect(lambda x: x)

    @ev.connect
    def on_my_event(sender, arg, kwarg=None):
        _list.append((arg, kwarg))

    ev.emit('my_event', ev, 'a')
    assert _list == [('a', None)]

    ev.emit('my_event', ev, 'b', 'c')
    assert _list == [('a', None), ('b', 'c')]

    ev.unconnect(on_my_event)

    ev.emit('my_event', ev, 'b', 'c')
    assert _list == [('a', None), ('b', 'c')]


def test_event_silent():
    ev = EventEmitter()

    _list = []

    @ev.connect()
    def on_test(sender, x):
        _list.append(x)

    ev.emit('test', ev, 1)
    assert _list == [1]

    with ev.silent():
        ev.emit('test', ev, 1)
    assert _list == [1]

    ev.set_silent(True)


def test_event_single():
    ev = EventEmitter()

    l = []

    @ev.connect(event='test')
    def on_test_bou(sender):
        l.append(0)

    @ev.connect  # noqa
    def on_test(sender):
        l.append(1)

    ev.emit('test', ev)
    assert l == [0, 1]

    ev.emit('test', ev, single=True)
    assert l == [0, 1, 0]
