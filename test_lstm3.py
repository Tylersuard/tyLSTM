import torch

def test_concatenate_hidden_and_word():
    from lstm3 import concatenate_hidden_and_word
    hidden_state_vector = torch.tensor([1,2,3])
    word_vector = torch.tensor([4])
    concatenated_tensor = concatenate_hidden_and_word(hidden_state_vector, word_vector)
    assert len(concatenated_tensor.tolist()) == 4

def test_create_gatenets():
    from lstm3 import VariableActivationFunctionNet
    forgetgatenet = VariableActivationFunctionNet(100, 39, "sigmoid")
    remembergatenet = VariableActivationFunctionNet(100, 39, "sigmoid")
    outputgatenet = VariableActivationFunctionNet(100, 39,"sigmoid")
    candidate_values_net = VariableActivationFunctionNet(100, 39, "tanh")
    assert forgetgatenet is not None
    assert remembergatenet is not None
    assert outputgatenet is not None
    assert candidate_values_net is not None

def test_create_lstm():
    from lstm3 import tyLSTM

    lstm = tyLSTM(word_vector=torch.tensor([1,2,3,4]),
                hidden_state_vector=torch.tensor([1,2,3,4]),
                cell_state_vector=torch.tensor([1,2,3,4]))
    assert lstm is not None

def test_create_forgotten_cell_state_vector():
    from lstm3 import tyLSTM
    lstm = tyLSTM(torch.tensor([1,2,3,4],dtype=torch.float32),torch.tensor([1,2,3,4],dtype=torch.float32),torch.tensor([1,2,3,4],dtype=torch.float32))
    forgotten_cell_state_vector = lstm.create_forgotten_cell_state_vector()
    assert forgotten_cell_state_vector is not None

def test_create_weighted_input_vector():
    from lstm3 import tyLSTM
    lstm = tyLSTM(torch.tensor([1,2,3,4],dtype=torch.float32),torch.tensor([1,2,3,4],dtype=torch.float32),torch.tensor([1,2,3,4],dtype=torch.float32))
    weighted_input_vector = lstm.create_weighted_input_vector()
    assert weighted_input_vector is not None

def test_create_new_cell_state_vector():
    from lstm3 import tyLSTM
    lstm = tyLSTM(torch.tensor([1,2,3,4],dtype=torch.float32),torch.tensor([1,2,3,4],dtype=torch.float32),torch.tensor([1,2,3,4],dtype=torch.float32))
    new_cell_state_vector = lstm.create_new_cell_state_vector()
    assert new_cell_state_vector is not None

def test_create_new_hidden_state_vector():
    from lstm3 import tyLSTM
    lstm = tyLSTM(torch.tensor([1,2,3,4],dtype=torch.float32),
                  torch.tensor([1,2,3,4],dtype=torch.float32),
                  torch.tensor([1,2,3,4],dtype=torch.float32))
    new_cell_state_vector = lstm.create_new_cell_state_vector()
    new_hidden_state_vector = lstm.create_new_hidden_state_vector(new_cell_state_vector)
    assert new_hidden_state_vector is not None

def test_predict_output_word_vector():
    from lstm3 import tyLSTM
    lstm = tyLSTM(
        torch.tensor([1,2,3,4],dtype=torch.float32),
        torch.tensor([1,2,3,4],dtype=torch.float32),
        torch.tensor([1,2,3,4],dtype=torch.float32))
    new_cell_state_vector = lstm.create_new_cell_state_vector()
    new_hidden_state_vector = lstm.create_new_hidden_state_vector(new_cell_state_vector)
    print(new_hidden_state_vector)
    output_word_vector = lstm.predict_output_word_vector(new_hidden_state_vector)
    assert output_word_vector is not None

