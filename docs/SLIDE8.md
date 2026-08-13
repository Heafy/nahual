# Slide 8 — Machine Learning · Random Forest 

<!-- FINAL SCRIPT: -->

## What's a decision tree

A decision tree is a supervised learning algorithm used for classification and regression tasks.
Consists of a root node, branches and leaf nodes. 

A decision tree split the dataset based on feature values to create pure subsets of the items in
a group that belongs to the same class. Asking different questions. 
For example:
1. Root node: Is the index fingertip higher than this line?
    Yes: Pass to internal node
    No: Go to a different question
2. Internal node:  Is the thumb touching the middle finger?
   Yes: It's letter b

The three builds itself during training, every questions thries thousand of possibilities and 
keeps the ones that best separates the letters that still in play until it ends on a single answer and 
it's how you would describe a hand sign: Thumb up, index and middle extended, that fingers are together. 
A tree learns the same kind or rule just measured instead of described.

But a single tree only memorize, give it 200 photos and it learn your hand, then fails on someone else.
Fix it using 200 decision trees and make every one of them use random features on purpose. Each tree doesn't look all the features at once. It picks a few random on how to split the data. So the trees stay different from each other.

Now each three making his own predition based on what it learned from its part of the data. And each 
single vote create confidencence. The forest answer independently and the system averages their answers.
If 190 trees saying "B" is high confidence around 95% of them. That confidence show how much the trees agree.I set the confidence threshold at 65% agreement for static letters, and 40% for the moving ones — motion is harder to pin down frame by frame, so I gave it more room.

Why this model, because it trains on seconds on a Macbook, no GPU, no cloud resources, it can run fast
enough for live video in with low device requirements and it's inspectable so when two letters get confused I can see exactly which pair and why.


<!-- BASE KNOWLEDGE: -->

## 


▎ One tree is basically a game of 20 questions. A decision tree asks yes/no questions, one at a time: is the index fingertip higher than this line, yes. Is the thumb touching the middle finger, no. Then it's the letter B. The tree builds itself during training, at every question it tries thousands of possible cutoffs and keeps the one that best separates the letters still in play, until each branch ends on a single answer. This fits sign language unusually well, because it's how a human would describe a letter out loud: thumb tucked, index and middle extended, fingers together. The tree learns the same kind of rule, just measured instead of described.

A single tree memorizes, give it 200 photos of your hand and it learns your hand, noise included, then fails on someone else's. The fix is to grow 200 trees and make every one of them slightly ignorant on purpose: each tree gets a different random sample of the training data, and each question is chosen from a small random handful of the available measurements, not all of them. That second rule matters most, the measurements overlap heavily, so without the restriction all 200 trees would latch onto the same obvious clue and become 200 copies of each other. Forcing some trees to work with a weaker clue makes them fail in different ways, and different mistakes cancel out when you average them. There's one more knob: the training data isn't balanced, so rarer letters get weighted up, otherwise the model would quietly learn to ignore them.

▎ The prediction is a vote, and the vote is the confidence. Show the forest  answer independently, and the system averages their answers. 190 treessaying "B" is high confidence, around 0.95. 80 saying "B", 70 saying "D", the rest scattered, is low confidence, around 0.4. That confidence number the demo displays is literally how much the trees agree, nothing extra had to be built for it. r: static letters turn red on screen below 0.65, motion letters get thrownaway entirely below 0.4 rather than shown as a wrong guess, since disagreement among the trees is a reliable sign the hand is mid-transition or in an ambiguous pose.            
▎ Why this model, and where it stops. It trains in seconds on a laptop, no GPU, no cloud, runs fast enough for live video, and it's inspectable, so when two letters get confused I can see exactly which pair and why. It also works with only tens of examplrs since every sample was recorded by hand. The cost: it can only reasonabout measurements I thought to give it, if a letter depends on something not in those 81 or 521 numbers, no amount of training fixes that, and it does poorly on poses far outside what was recorded. A neural network could learn the motion featurehand-designed part entirely, but it would need far more data, far moretraining time, and wouldn't hand me an explanation of why it decided anything.
▎
▎ The line that lands: 200 trees, each deliberately given an incomplete view, voting. Their level of agreement is the confidence score on screen.