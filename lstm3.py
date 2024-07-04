import torch
import torch.nn as nn

with open("text_corpus.txt", "r", encoding="utf-8") as readfile:
    text = readfile.read()

text = text.lower()
all_unique_characters = sorted(list(set(text)))
print(len(all_unique_characters))
print(all_unique_characters)

def vectorize_one_character(character):
    char_vector_list = [1 if character == char else 0 for char in all_unique_characters]
    return torch.unsqueeze(torch.tensor(char_vector_list, dtype=torch.float32), 0)

def unvectorize_one_vector(character_vector):
    index = torch.argmax(character_vector).item()
    return all_unique_characters[index]

def concatenate_hidden_and_word(hidden_state_vector, word_vector):
    # print(hidden_state_vector.shape)
    # print(word_vector.shape)
    return torch.cat((hidden_state_vector, word_vector),dim=1)

class VariableActivationFunctionNet(nn.Module):
    def __init__(self, len_hidden_state_vector, len_word_vector, activation_function):
        super().__init__()
        self.fc1 = nn.Linear(len_hidden_state_vector+len_word_vector, len_hidden_state_vector)
        # print(len_hidden_state_vector+len_word_vector)
        # print(len_hidden_state_vector)
        self.activation_function = activation_function
    
    def forward(self, x):
        if self.activation_function.lower() == "sigmoid":
            return torch.sigmoid(self.fc1(x))
        elif self.activation_function.lower() == "tanh":
            return torch.tanh(self.fc1(x))
        
class HiddenStateVectorToPredictedWordVectorNet(nn.Module):
    def __init__(self, len_hidden_state_vector, len_word_vector):
        super().__init__()
        self.fc1 = nn.Linear(len_hidden_state_vector, len_word_vector)

    def forward(self, x):
        return self.fc1(x)
        
class tyLSTM(nn.Module):
    def __init__(self, word_vector, hidden_state_vector, cell_state_vector):
        super().__init__()

        self.word_vector = word_vector
        self.hidden_state_vector = hidden_state_vector
        self.cell_state_vector = cell_state_vector

        len_word_vector = self.word_vector.shape[1]
        len_hidden_state_vector = self.hidden_state_vector.shape[1]

        self.forgetgatenet = VariableActivationFunctionNet(
            len_hidden_state_vector=len_hidden_state_vector, 
            len_word_vector=len_word_vector, 
            activation_function="sigmoid")

        self.remembergatenet = VariableActivationFunctionNet(
            len_hidden_state_vector=len_hidden_state_vector, 
            len_word_vector=len_word_vector, 
            activation_function="sigmoid")
        
        self.outputgatenet = VariableActivationFunctionNet(
            len_hidden_state_vector=len_hidden_state_vector, 
            len_word_vector=len_word_vector, 
            activation_function="sigmoid")
        
        self.candidatevaluesnet = VariableActivationFunctionNet(
            len_hidden_state_vector=len_hidden_state_vector, 
            len_word_vector=len_word_vector, 
            activation_function="tanh")
        
        self.hidden2outputwordvectornet = HiddenStateVectorToPredictedWordVectorNet(
            len_hidden_state_vector=len_hidden_state_vector, 
            len_word_vector=len_word_vector)
        
    def create_forgotten_cell_state_vector(self):
        input = concatenate_hidden_and_word(self.hidden_state_vector, self.word_vector)
        output = self.forgetgatenet(input)
        forgotten_cell_state_vector = output * self.cell_state_vector
        return forgotten_cell_state_vector
    
    def create_weighted_input_vector(self):
        input = concatenate_hidden_and_word(self.hidden_state_vector, self.word_vector)
        remember_gate = self.remembergatenet(input)
        candidate_values = self.candidatevaluesnet(input)
        weighted_input_vector = remember_gate * candidate_values
        return weighted_input_vector
    
    def create_new_cell_state_vector(self):
        forgotten_cell_state_vector = self.create_forgotten_cell_state_vector()
        weighted_input_vector = self.create_weighted_input_vector()
        new_cell_state_vector = forgotten_cell_state_vector + weighted_input_vector
        return new_cell_state_vector
    
    def create_new_hidden_state_vector(self, new_cell_state_vector):
        input = concatenate_hidden_and_word(self.hidden_state_vector, self.word_vector)
        output_gate = self.outputgatenet(input)
        scaled_new_cell_state_vector = torch.tanh(new_cell_state_vector)
        new_hidden_state_vector = output_gate * scaled_new_cell_state_vector
        return torch.tensor(new_hidden_state_vector)

    def predict_output_word_vector(self, new_hidden_state_vector):
        output_word_vector = self.hidden2outputwordvectornet(new_hidden_state_vector)
        return output_word_vector
    
    def forward(self, word_vector, hidden_state_vector, cell_state_vector):
        self.word_vector = word_vector
        self.hidden_state_vector = hidden_state_vector
        self.cell_state_vector = cell_state_vector

        new_cell_state_vector = self.create_new_cell_state_vector()
        new_hidden_state_vector = self.create_new_hidden_state_vector(new_cell_state_vector)
        output_word_vector = self.predict_output_word_vector(new_hidden_state_vector)

        return output_word_vector, new_hidden_state_vector, new_cell_state_vector

def train(X, X_test):


    #Process dataset and create a word vector
    letter = "a"
    example_word_vector = vectorize_one_character(letter)
    cell_state_vector = torch.randn(1, 100)
    hidden_state_vector = torch.randn(1, 100)


    tylstm = tyLSTM(
        example_word_vector, 
        hidden_state_vector,
        cell_state_vector)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(tylstm.parameters(), lr = 0.02)
    
    epochs = 3
    total_loss=0
    for epoch in range(epochs):
        for i in range(len(X)-1):
            optimizer.zero_grad()

            input_word_vector = vectorize_one_character(X[i])
            expected_output_word_vector = vectorize_one_character(X[i+1])
            (output_word_vector, 
             new_hidden_state_vector, 
             new_cell_state_vector) = tylstm(input_word_vector,
                                                    hidden_state_vector,
                                                    cell_state_vector)
            
            hidden_state_vector = new_hidden_state_vector
            cell_state_vector = new_cell_state_vector

            loss = criterion(output_word_vector, expected_output_word_vector)
            loss.backward(retain_graph=True)
            optimizer.step()

            total_loss += loss.item()

            if i % 100 == 0:
                average_loss = total_loss / (i+1)
                print(f' Epoch {epoch+1}/{epochs}, Step {i}/{len(X)}, Average Loss: {average_loss}')
    return tylstm

train_test_split = int(len(text) * .9)
trained_model = train(text[:train_test_split], text[train_test_split:])