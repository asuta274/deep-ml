import numpy as np

def rnn_forward(input_sequence: list[list[float]], initial_hidden_state: list[float], Wx: list[list[float]], Wh: list[list[float]], b: list[float]) -> list[float]:
	initial_hidden_state = np.array(initial_hidden_state)
	Wx = np.array(Wx)
	Wh = np.array(Wh)
	b = np.array(b)
	input_sequence = np.array(input_sequence)
	
	n = len(input_sequence)
	hidden_state = initial_hidden_state
	for t in range(n):
		hidden_state = np.tanh(Wx @ input_sequence[t] + Wh @ hidden_state + b)
	return np.round(hidden_state, 4)