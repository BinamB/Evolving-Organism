# Evolving-Organism
  This virtual organism runs using a genetic algorithm and a neural network. There were three different food we wanted to introduce: poisonous food(green), healthy food(blue) and superfood(red). The poisonous food decreases the organisms score by 1, healthy food increases the score by 1 and superfood increases the score by 5.
The basis of this program was Evolving Simple Organism using a [Genetic Algorithm and Deep Learning from Scratch with Python by Nathan Rooy](https://nathanrooy.github.io/posts/2017-11-30/evolving-simple-organisms-using-a-genetic-algorithm-and-deep-learning/). This organism only ate one thing. The idea was to make it learn how to avoid the poisonous food and prefer one type of food over the other. 
We made some additions to the base code and realized that when adding different types of food, the organism wasn’t “learning” anymore. We found that adding more hidden layers was the answer to the problem, so we changed from a single hidden layer to three hidden layers. Each hidden layer corresponds to a single type of food. 
In our case, regular food, superfood and poison. It contains one input layer which is the direction to the nearest food particle. Three hidden layers representing each food particle and two output layers which represents turning rate and change in velocity. Their fitness is determined by their velocity and turning speed. The fitter they are the faster they get. 

![alt text](https://github.com/BinamB/Evolving-Organism/blob/master/Capture.PNG)

We looked at different activation functions. We first started by using the tanh function that was originally being used. This worked well since its range is from [-1, 1] and gave us an average score of 20-30. We tried changing the activation function tanh to the sigmoid function.  The range of this function is [0, 1], which was not compatible with our outputs.
We wanted a sensitive function that ranges from [-1, 1], so we edited the sigmoid function to get our desired range. Doing this gave us almost the same average as the tanh function. Playing around with the functions a little more we realized that tanh(3x) was the way to go. If we look at the graph below we can see that the tanh(3x) (orange) function is the steepest which means a small 
change in x value brings large changes in the y value which we want 

![alt text](https://github.com/BinamB/Evolving-Organism/blob/master/tanh%20vs%20sigmoid%20functions.PNG)

Running our code with this activation function we get the following output. 

![alt text](https://github.com/BinamB/Evolving-Organism/blob/master/fitness.PNG)

Running this code for more generations yield similar average values.

![alt text](https://github.com/BinamB/Evolving-Organism/blob/master/sample.gif)



