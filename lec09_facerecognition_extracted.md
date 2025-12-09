## CSE 152B: Computer Vision II Manmohan Chandraker 

**==> picture [720 x 54] intentionally omitted <==**

## Lecture 9: Face Recognition 

**==> picture [198 x 149] intentionally omitted <==**

**==> picture [426 x 196] intentionally omitted <==**

## Course details 

• Class webpage: 

– http://cseweb.ucsd.edu/~mkchandraker/classes/CSE152B/Spring2025/ 

• Instructor email: 

   - mkchandraker@ucsd.edu 

- TA: Mustafa Yaldiz 

   - Emails: myaldiz@ucsd.edu 

• Grading 

- 40% assignments 

- 25% midterm (open notes) 

- 35% final exam (open notes) 

• Aim is to learn together, discuss and have fun! 

CSE 152B, SP25: Manmohan Chandraker 

Overall goals for the course 

• Introduce the state-of-the-art in computer vision 

• Study principles that make them possible 

• Get understanding of tools that drive computer vision 

• Enable one or all of several such outcomes 

- Pursue higher studies in computer vision 

- Join industry to do cutting-edge work in AI 

- Gain an appreciation of modern AI technologies 

# Recap 

CSE 152B, SP25: Manmohan Chandraker 

## Unconstrained Face Recognition 

**==> picture [356 x 217] intentionally omitted <==**

**==> picture [216 x 217] intentionally omitted <==**

**==> picture [714 x 146] intentionally omitted <==**

CSE 152B, SP25: Manmohan Chandraker 

[Liao et al., Partial Face Recognition, PAMI 2013] 

## Face Recognition on LFW Benchmark 

**==> picture [549 x 162] intentionally omitted <==**

• Human performance : **99.20%** • Local Binary Patterns : 95.17% • DeepFace : 97.35 % • DeepID2 : 99.15% • FaceNet : **99.63%** 

**==> picture [188 x 146] intentionally omitted <==**

**==> picture [193 x 138] intentionally omitted <==**

CSE 152B, SP25: Manmohan Chandraker 

## Face Identification 

- Closed set identification: assign one of gallery identities to probe image 

- • Galleries can be very large, high chance of similar appearances 

- Goal is to have sharp decision boundary between gallery identities 

- Feature need not generalize to other tasks (identities outside the gallery) 

**==> picture [667 x 284] intentionally omitted <==**

**----- Start of picture text -----**<br>
Face Identification Tom Cruise<br>Probe<br>Gallery<br>CSE 152B, SP25: Manmohan Chandraker<br>**----- End of picture text -----**<br>


## Face Verification 

- Given a pair of face images: • A squared L2 distance                  is used to determine same or different 

- • Good embedding: true matches will lie within a small value of 

**==> picture [166 x 249] intentionally omitted <==**

**==> picture [248 x 57] intentionally omitted <==**

Face 

Verification 

**==> picture [248 x 57] intentionally omitted <==**

**==> picture [95 x 141] intentionally omitted <==**

CSE 152B, SP25: Manmohan Chandraker 

Same 

## Different 

## Verification and Identification Signals 

**==> picture [100 x 30] intentionally omitted <==**

**----- Start of picture text -----**<br>
Verification<br>**----- End of picture text -----**<br>


**==> picture [114 x 30] intentionally omitted <==**

**----- Start of picture text -----**<br>
Identification<br>**----- End of picture text -----**<br>


- **Identification:** 

   - Distinguish images of one identity from another identity 

   - Favors large distance between clusters 

   - Stronger learning signal, but need not generalize to new identities 

- **Verification:** 

   - Match two images of an individual across large appearance variations 

   - • Favors tight clusters for each identity 

   - Weaker learning signal, but feature applicable to new identities 

## Steps in Face Recognition 

**==> picture [115 x 115] intentionally omitted <==**

- Face Detection 

   - Localize the face 

- Face Alignment 

– Factor out 3D transformation 

**==> picture [91 x 115] intentionally omitted <==**

**==> picture [177 x 107] intentionally omitted <==**

- Feature Extraction 

– Find compact representation 

• Classification 

- Answer the question 

## Challenges in Face Alignment 

- Infer 3D from 2D 

   - Slight occlusion 

   - Lighting condition 

   - Head orientation 

   - Non rigid deformation 

**==> picture [289 x 215] intentionally omitted <==**

## DeepFace Alignment: Substep 1 

• 2D feature point extraction 

- 𝑥 

- 2D alignment !"#$%& = (𝑆∗𝑅∗𝑇)𝑥'%(&#) 

- • Only for in plane alignment 

**==> picture [121 x 143] intentionally omitted <==**

**==> picture [112 x 121] intentionally omitted <==**

**==> picture [111 x 121] intentionally omitted <==**

## Fiducial Point Detection 

## 2D Transformation 

## Until convergence 

[Taigman et al., DeepFace] 

CSE 152B, SP25: Manmohan Chandraker 

# DeepFace Alignment: Substep 2 

# • 3D feature point extraction 

• 3D alignment 

Reference 3D Fiducial Point Location min 𝑟[!] Σ["#] 𝑟 𝑟= 𝑥 $% −𝑃𝑥&% Detected 2D Final Fiducial Point Location Alignment 

Detected 2D Fiducial Point Location 

[Taigman et al., DeepFace] 

## Architecture 

**==> picture [698 x 158] intentionally omitted <==**

**==> picture [620 x 129] intentionally omitted <==**

[Taigman et al., DeepFace] 

## Architecture 

**==> picture [698 x 158] intentionally omitted <==**

**==> picture [620 x 129] intentionally omitted <==**

- Different regions of an aligned image have different local statistics 

   - Aligned images with similar semantic concepts are being considered 

   - A large training dataset is available, can handle increased parameters 

[Taigman et al., DeepFace] 

CSE 152B, SP25: Manmohan Chandraker 

## Architecture 

**==> picture [698 x 158] intentionally omitted <==**

**==> picture [666 x 74] intentionally omitted <==**

**==> picture [61 x 26] intentionally omitted <==**

**==> picture [391 x 26] intentionally omitted <==**

**==> picture [87 x 26] intentionally omitted <==**

**==> picture [603 x 33] intentionally omitted <==**

Target probability distribution _pi_ = 1 for class _t_ and 0 for other _i_ 

Predicted probability distribution 

[Taigman et al., DeepFace] 

CSE 152B, SP25: Manmohan Chandraker 

## Entropy 

• A measure of randomness in a system 

– The higher the entropy, the less ordered is the system 

## Entropy 

• A measure of randomness in a system 

– The higher the entropy, the less ordered is the system 

– Natural systems tend to assume a state of higher entropy 

CSE 152B, SP25: Manmohan Chandraker 

## Entropy in Classification 

- Suppose we are doing 5-way classification 

- Some of these output distributions are better than others 

**==> picture [697 x 177] intentionally omitted <==**

CSE 152B, SP25: Manmohan Chandraker 

## Entropy in Classification 

- Suppose we are doing 5-way classification 

- Some of these output distributions are better than others 

**==> picture [697 x 177] intentionally omitted <==**

• More peaky distribution is better for classification 

- More peaky distribution = Lower entropy 

- More uncertainty = Higher entropy 

CSE 152B, SP25: Manmohan Chandraker 

## Information Content 

- Consider an unfair coin with p(Heads) = 0.99 

   - A coin toss that yields H is not surprising 

   - But a toss that yields T is very surprising 

- Information content of a stochastic event E 

**==> picture [270 x 34] intentionally omitted <==**

   - Logarithm in base 2: information in bits 

   - Natural logarithm: information in nats 

- For unfair coin, 

   - Information in event Heads: –log(0.99) = 0.01 bits 

   - Information in event Tails: -log(0.01) = 6.64 bits 

   - Matches intuition for Tails being a more surprising event than Heads 

## Entropy 

• Entropy: expected rate of information from stochastic process 

- For a random variable X, expected value is 

– When the random variable is information, expectation is 

**==> picture [453 x 59] intentionally omitted <==**

- For the unfair coin with p(H) = 0.99, we have 

CSE 152B, SP25: Manmohan Chandraker 

## Entropy 

• Entropy: expected rate of information from stochastic process 

- For a random variable X, expected value is 

- When the random variable is information, expectation is 

**==> picture [453 x 59] intentionally omitted <==**

- For the unfair coin with p(H) = 0.99, we have 

- For a fair coin with p(Heads) = 0.5, we have 

𝐻(𝑋) = −(0.5 𝑙𝑜𝑔(0.5) + 0.5 𝑙𝑜𝑔(0.5)) = 1 𝑏𝑖𝑡 

- CSE 152B, SP25: Manmohan Chandraker 

## Entropy 

• Entropy: expected rate of information from stochastic process 

- For a random variable X, expected value is 

- When the random variable is information, expectation is 

**==> picture [453 x 59] intentionally omitted <==**

- For the unfair coin with p(H) = 0.99, we have 

- For a fair coin with p(Heads) = 0.5, we have 

𝐻(𝑋) = −(0.5 𝑙𝑜𝑔(0.5) + 0.5 𝑙𝑜𝑔(0.5)) = 1 𝑏𝑖𝑡 

• ____ uncertainty = _____ entropy 

- CSE 152B, SP25: Manmohan Chandraker 

## Entropy 

• Entropy: expected rate of information from stochastic process 

- For a random variable X, expected value is 

- When the random variable is information, expectation is 

**==> picture [453 x 59] intentionally omitted <==**

- For the unfair coin with p(H) = 0.99, we have 

- For a fair coin with p(Heads) = 0.5, we have 

𝐻(𝑋) = −(0.5 𝑙𝑜𝑔(0.5) + 0.5 𝑙𝑜𝑔(0.5)) = 1 𝑏𝑖𝑡 

- More uncertainty = Higher entropy 

   - The unfair coin delivers very little information (mostly just Heads) 

CSE 152B, SP25: Manmohan Chandraker 

## Cross-Entropy and Softmax 

**==> picture [580 x 143] intentionally omitted <==**

**----- Start of picture text -----**<br>
High entropy Low entropy<br>Labels Labels<br>Prediction Prediction<br>**----- End of picture text -----**<br>


- Entropy: optimal way to transmit information about an event 

• Cross-entropy: distance between ground truth and predicted distributions 

**==> picture [248 x 46] intentionally omitted <==**

**==> picture [355 x 114] intentionally omitted <==**

• Softmax loss: cross-entropy on 

softmax probabilities 

CSE 152B, SP25: Manmohan Chandraker 

## DeepID2: Combine Identification and Verification 

**==> picture [682 x 186] intentionally omitted <==**

- Locally connected layer at the top 

   - Respond to facial features at preferred spatial locations 

- Feature layer fully connected to last convolutional and locally connected layer 

   - Multiscale information 

   - Face representation: 

**==> picture [138 x 24] intentionally omitted <==**

[Sun et al., DeepID2] 

## Identification Signal 

**==> picture [540 x 148] intentionally omitted <==**

- _n_ 

- Identification: connect feature layer to -way softmax layer 

   - Outputs a probability distribution over _n_ classes 

   - Train with a cross-entropy loss 

**==> picture [390 x 63] intentionally omitted <==**

Feature 

Target class 

Target probability distribution Predicted probability _pi_ = 1 for class _t_ and 0 for other _i_ distribution 

- Goal is to correctly classify all identities simultaneously 

   - Incentivize learning discriminative features across inter-personal variations [Sun et al., DeepID2] 

   - CSE 152B, SP25: Manmohan Chandraker 

[Sun et al., DeepID2] 

## Verification Signal 

**==> picture [540 x 148] intentionally omitted <==**

• Verification: directly regularize the feature vector 

• Pairwise: Gather faces from same class, push those from different classes 

**==> picture [524 x 63] intentionally omitted <==**

• Cosine similarity: 

**==> picture [340 x 43] intentionally omitted <==**

**==> picture [125 x 34] intentionally omitted <==**

**==> picture [85 x 24] intentionally omitted <==**

• Goal is to learn features that can be matched across intra-personal variations 

CSE 152B, SP25: Manmohan Chandraker 

[Sun et al., DeepID2] 

## Balancing Identification and Verification 

**==> picture [599 x 234] intentionally omitted <==**

Dataset mean 

Class _i_ mean 

**==> picture [275 x 34] intentionally omitted <==**

_c_ classes, _ni_ instances in class _i_ 

- Inter-class scatter : 

**==> picture [312 x 46] intentionally omitted <==**

- Intra-class scatter : 

- Variance in scatter indicated by size of eigenvalues 

- Small number of eigenvectors: diversity of variation is low 

- Both diversity and magnitude of feature variance matters for recognition 

CSE 152B, SP25: Manmohan Chandraker 

## Balancing Identification and Verification 

**==> picture [599 x 234] intentionally omitted <==**

**----- Start of picture text -----**<br>
Only verification Only verification<br>Only identification Only identification<br>**----- End of picture text -----**<br>


- When only identification signal is used                 : 

   - High diversity in both inter-personal and intra-personal features 

   - Good for identification since it helps distinguish different identities 

   - But large intra-personal variance is noise for verification 

## Balancing Identification and Verification 

**==> picture [599 x 234] intentionally omitted <==**

**----- Start of picture text -----**<br>
Only verification Only verification<br>Only identification Only identification<br>**----- End of picture text -----**<br>


- When only identification signal is used                 : 

   - High diversity in both inter-personal and intra-personal features 

   - Good for identification since it helps distinguish different identities 

   - But large intra-personal variance is noise for verification 

- When only verification signal is used (     approaches        ) : • Both intra-personal and inter-personal variance collapse to few directions 

   - Cannot distinguish between different identities CSE 152B, SP25: Manmohan Chandraker 

## Balancing Identification and Verification 

**==> picture [599 x 234] intentionally omitted <==**

**----- Start of picture text -----**<br>
Only verification Only verification<br>Only identification Only identification<br>**----- End of picture text -----**<br>


- When both verification and identification signals are used                      : • Inter-personal variations stay high 

   - Intra-personal variations reduce in diversity and magnitude 

## Balancing Identification and Verification 

**==> picture [690 x 183] intentionally omitted <==**

- Visualize features for 6 identities 

- With only identification signal: • Cluster centers are well-separated, but large cluster size causes overlap 

- With only verification signal: 

   - Cluster sizes become small, but cluster centers also collapse 

- With both signals : 

   - Clusters sizes become small and cluster centers are reasonably separated CSE 152B, SP25: Manmohan Chandraker 

## Learn an Embedding for Face Recognition 

- **Face verification :** determine whether two images are of the same person 

- **Face identification :** determine identity of person in an image 

- **Face clustering :** find the same person among a collection of faces 

- Train a network such that embedding distances directly represent similarity • Faces of same person : small distances 

   - Faces of different persons : large distances 

- Once embedding is learned, above problems are all solvable 

   - **Verification :** threshold distance between two embeddings 

   - **Identification :** can be posed as k-NN classification 

   - **Clustering :** can be solved using methods like k-means 

## FaceNet: Learn an Embedding for Face Recognition 

**==> picture [442 x 79] intentionally omitted <==**

- Goal: learn _d_ -dimensional embedding _f(x)_ for face image _x_ 

- • Constrain embedding to lie on unit sphere: 

- Goal for triplet loss: 

   - Minimize distance between anchor and a positive (from same class) 

   - Maximize distance between anchor and a negative (from different classes) 

**==> picture [501 x 119] intentionally omitted <==**

## Triplet Loss for Training 

- Goal for triplet loss: 

   - Minimize distance between anchor image       and a positive 

**==> picture [23 x 27] intentionally omitted <==**

- Maximize distance between anchor        and a negative 

**==> picture [29 x 27] intentionally omitted <==**

**==> picture [410 x 98] intentionally omitted <==**

**==> picture [366 x 31] intentionally omitted <==**

, 

**==> picture [243 x 27] intentionally omitted <==**

• Total loss to minimize: 

**==> picture [411 x 62] intentionally omitted <==**

- Challenge: too many triplets satisfy the margin easily 

- • Need to select hard examples that are active and improve the model 

CSE 152B, SP25: Manmohan Chandraker 

## Triplet Selection 

• To ensure fast convergence, given an anchor       : 

**==> picture [247 x 32] intentionally omitted <==**

• Select **hardest positive** such that 

**==> picture [241 x 32] intentionally omitted <==**

• Select **hardest negative** such that 

**Hard Negative Hard Positive Negative Space Embedding Positive Space location of anchor** 

## Triplet Selection 

• To ensure fast convergence, given an anchor       : 

**==> picture [247 x 32] intentionally omitted <==**

- Select **hardest positive** such that 

**==> picture [241 x 32] intentionally omitted <==**

- Select **hardest negative** such that 

**==> picture [473 x 224] intentionally omitted <==**

**----- Start of picture text -----**<br>
Hard<br>Negative<br>Hard<br>Positive<br>Negative Space<br>Embedding<br>Positive Space<br>location of<br>anchor<br>**----- End of picture text -----**<br>


• **Inefficient** (or infeasible) to compute argmin and argmax over training set 

• Might lead to **poor training** as mislabeled or poorly imaged examples dominate 

CSE 152B, SP25: Manmohan Chandraker 

## Triplet Selection 

## • Two courses of action: offline and online selection of triplets 

## • **Offline:** every n steps, use current feature for argmin and argmax on subset 

## Triplet Selection 

- Two courses of action: offline and online selection of triplets 

- **Offline:** every n steps, use current feature for argmin and argmax on subset 

- **Online:** selecting hard positive and negative examples in mini-batch 

   - Use large mini-batch with several thousand examples 

   - Use several examples per identity for meaningful anchor-positive distances 

   - • Randomly sample negatives from other identities 

## Triplet Selection 

**==> picture [295 x 266] intentionally omitted <==**

**----- Start of picture text -----**<br>
n<br>n<br>n<br>n<br>n<br>p<br>n<br>p<br>n p<br>n<br>n<br>n<br>n n<br>**----- End of picture text -----**<br>


- In practice: 

   - Use all anchor-positive pairs, instead of just hard positives 

   - Use **semi-hard negatives** at beginning of training 

**==> picture [287 x 32] intentionally omitted <==**

- Semi-hard negatives: can be within margin, but further than positives 

Strictly hard negatives at beginning can cause feature to collapse to _f(x)_ = 0 

- 

CSE 152B, SP25: Manmohan Chandraker 

## Comparison of DeepFace, DeepID2, FaceNet 

**==> picture [645 x 169] intentionally omitted <==**

- Benefit over DeepFace: 

   - Learns an embedding, can be used for multiple tasks 

   - Only 128-dimensional representation, efficient for inference 

## Comparison of DeepFace, DeepID2, FaceNet 

**==> picture [645 x 169] intentionally omitted <==**

• Benefit over DeepFace: 

• Learns an embedding, can be used for multiple tasks 

- Only 128-dimensional representation, efficient for inference 

• Intuitive benefit of triplet loss over pairwise loss in DeepID2 

- Pairwise: map all faces from one identity to same point 

• Triplet: margin between each pair of faces of an identity and all other faces 

• Triplet loss allows an identity manifold, with distance from other identities CSE 152B, SP25: Manmohan Chandraker 

