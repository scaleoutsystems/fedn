# state machine proposal for handling events at combiner
#


class State:

    def event(self, event):
        assert 0

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return self.__class__.__name__


class InitialState(State):

    def event(self, event):
        if event == 'instruct':
            return InstructingState()

        return self


class InstructingState:

    def event(self, event):
        if event == 'error':
            return InitialState()
        elif event == 'combine':
            return CombineState()

        return self


class CombineState:

    def event(self, event):
        if event == 'error':
            return InstructingState()
        elif event == 'finish':
            return InitialState()

        return self
