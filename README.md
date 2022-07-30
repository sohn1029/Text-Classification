# Text Classification with GRU and attention
The model to be introduced this time is for text classification.
Text classification is the operation in which text is input and the type of text is output. 
To do this, we need data that the class is labelled, a model for NLP.
## Data Preview
Data is received from a client, not directly collected or processed. The appearance is as follows.    

![data_preview](https://user-images.githubusercontent.com/31722713/181917203-86d542f2-8989-4476-aeac-bcecb5ffc09e.png)

## Model
RNN, LSTM, and GRU are widely used as models for natural language processing.
In particular, unlike RNN, LSTM and GRU are mainly used because they remember past contents well.
In this project, GRU was used because GRU performed better than LSTM.
However, the output of the GRU alone lacked performance.
So, I came up with several strategies to solve this problem.  
- First, pass the text not only in the forward direction but also in the reverse direction to the GRU.
I thought that if reverse information was also available, information lost in the forward direction could be supplemented.  
- Second, I thought it would be good to refer to all the outputs of the LSTM.
The existing code made predictions with only the last output, but I thought I could improve the performance by using the remaining outputs.  
- Finally, I thought it would be good to add a little more to the second so that I can attach importance to the output and watch more important parts.  

To solve this problem, the bidirection option of the torch GRU and attention technique were used.
An overview of this project model is as follows.

![model](https://user-images.githubusercontent.com/31722713/181918264-d5f25b4f-f76c-4d3c-9845-f5cd231df651.png)

## Result
The training results are as follows.

![model_acc](https://user-images.githubusercontent.com/31722713/181918741-5242dd3b-10c6-4431-9ee2-04ef77d2c0c1.png)

The validation accuracy is 81%.
And I also visualized whether the attention array works properly.

![attn1](https://user-images.githubusercontent.com/31722713/181919288-945b6eaf-aeeb-420d-8974-0ad9ec77bf41.png)
![attn2](https://user-images.githubusercontent.com/31722713/181919352-c67feab8-c3d5-4a4a-a291-098fe3fe20a0.png)

Looking at the figure above, it can be seen that model focused a little more on special words with high numbers.  

