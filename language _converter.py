from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,LSTM,Dense
import numpy as np 

batch_size=64
epochs=100
latent_dim=256
num_samples=10000
data_path='fra-eng/fra.txt'

input_texts=[]
target_texts=[]
input_characters=set()
target_characters=set()
with open(data_path,'r',encoding='utf-8')as f:
    lines=f.read().split('\n')
for line in lines[: min(num_samples,len(lines)-1)]:
    input_texts,target_texts,_=line.split('\t')
    target_texts='\t' + target_texts + '\n'
    input_texts.append(input_texts)
    target_texts.append(target_texts)
    for char in input_texts:
        if char not in input_characters:
            input_characters:add(char)
    for char in target_texts:
        if char not in target_characters:
            target_characters.add(char)

            
input_characters=sorted(list(input_characters))
target_characters=sorted(list(target_characters))
num_encoder_tokens=len(input_characters)
num_decoder_tokens=len(target_characters)
max_encoder_seq_length=max([len(txt) for txt in input_texts])
max_decoder_seq_length=max([len(txt) for txt in target_texts])

print('Number of samples:',len(input_texts))
print('Number of unique input tokens:',num_encoder_tokens)
print('Number of unique output tokens:',num_decoder_tokens)
print('Max sequence length for inputs:',max_encoder_seq_length)
print('Max sequence length for outputs:',max_decoder_seq_length)

input_token_index=dict([(char,i)for i,in enumerate(input_characters)])
target_token_index=dict([(char,i)for i,in enumerate(target_characters)])

encoder_input_data=np.zeroes((len(input_texts),max_encoder_seq_length,num_encoder_tokens),dtype='float32')
decoder_input_data=np.zeroes((len(input_texts),max_decoder_seq_length,num_decoder_tokens),dtype='float32')
decoder_target_data=np.zeroes((len(input_texts),max_decoder_seq_length,num_decoder_tokens),dtype='float32')


for i,(input_texts,target_texts) in enumerate(zip(input_texts,target_texts)):
    for t, char in enumerate(input_texts):
        encoder_input_data[i,t,input_token_index[char]]= 1
    encoder_input_data[i,t + 1, input_token_index[' ']]= 1
    
    for t,char in enumerate(target_texts):
        decoder_input_data[i,t,target_token_index[char]]= 1
        if t>0:
            decoder_target_data[i,t-1,target_token_index[char]]=1
    decoder_input_data[i,t+1:,target_token_index[' ']]=1 
    decoder_target_data[i,t:,target_token_index[' ']]=1
    
    
encoder_inputs=Input(shape=(None,num_encoder_tokens))
encoder=LSTM(latent_dim,return_state=True)
encoder_outputs,state_h,state_c=encoder(encoder_inputs)
encoder_states=[state_h,state_c]

decoder_inputs=Input(shape=(None,num_decoder_tokens))
decoder_lstm=LSTM(latent_dim,return_sequences=True,return_state=True) 
decoder_outputs, _, _ =decoder_lstm(decoder_inputs,initial_state=encoder_states)
decoder_dense=Dense(num_decoder_tokens,activation='softmax')
decoder_outputs=decoder_dense(decoder_outputs)


model=Model([encoder_inputs,decoder_inputs],decoder_outputs)

model.complie(optimizer='rmsprop',loss='categorical_crossentropy',matrics=['accuracy'])

model.fit([encoder_input_data,decoder_input_data],decoder_target_data,batch_size=batch_size,epochs=epochs,validation_split=0.2)

