# DeBiasingWE
Debiasing Word Embeddings and removing gender biases from pretrained Glove Vectors.

# 1 - Cosine similarity

To measure how similar two words are, we need a way to measure the degree of similarity between two embedding vectors for the two words. Given two vectors ![eq4](http://latex.codecogs.com/gif.latex?u) and ![eq4](http://latex.codecogs.com/gif.latex?v), cosine similarity is defined as follows:

![eq1](http://latex.codecogs.com/gif.latex?CosineSimilarity%28u%2C%20v%29%20%3D%20%5Cfrac%20%7Bu%20.%20v%7D%20%7B%7C%7Cu%7C%7C_2%20.%20%7C%7Cv%7C%7C_2%7D%20%3D%20cos%28%5Ctheta%29)   
where ![eq2](http://latex.codecogs.com/gif.latex?%24u.v%24) is the dot product (or inner product) of two vectors, ![eq3](http://latex.codecogs.com/gif.latex?%7C%7Cu%7C%7C_2) is the norm (or length) of the vector ![eq4](http://latex.codecogs.com/gif.latex?u), and ![eq4](http://latex.codecogs.com/gif.latex?%24%5Ctheta%24) is the angle between ![eq4](http://latex.codecogs.com/gif.latex?u) and ![eq4](http://latex.codecogs.com/gif.latex?v). This similarity depends on the angle between ![eq4](http://latex.codecogs.com/gif.latex?u) and ![eq4](http://latex.codecogs.com/gif.latex?v). If ![eq4](http://latex.codecogs.com/gif.latex?u) and ![eq4](http://latex.codecogs.com/gif.latex?v) are very similar, their cosine similarity will be close to 1; if they are dissimilar, the cosine similarity will take a smaller value.

<img src="https://raw.githubusercontent.com/00arun00/DeBiasingWE/master/images/cosine_sim.png" style="width:800px;height:250px;">
<caption><center> **Figure 1**: The cosine of the angle between two vectors is a measure of how similar they are</center></caption>

**Reminder**: The norm of ![eq4](http://latex.codecogs.com/gif.latex?u) is defined as ![eq4](http://latex.codecogs.com/gif.latex?%7C%7Cu%7C%7C_2%20%3D%20%5Csqrt%7B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%20u_i%5E2%7D)


## 2 - Debiasing word vectors   
We can examine gender biases that can be reflected in a word embedding, and explore algorithms for reducing the bias.

Lets first see how the GloVe word embeddings relate to gender. First let us compute a vector ![eq](http://latex.codecogs.com/gif.latex?%24g%20%3D%20e_%7Bwoman%7D-e_%7Bman%7D%24), where ![eq](http://latex.codecogs.com/gif.latex?%24e_%7Bwoman%7D%24) represents the word vector corresponding to the word *woman*, and ![eq](http://latex.codecogs.com/gif.latex?%24e_%7Bman%7D%24) corresponds to the word vector corresponding to the word *man*. The resulting vector ![eq](http://latex.codecogs.com/gif.latex?%24g%24) roughly encodes the concept of "gender". (we can get a more accurate representation if we compute ![eq](http://latex.codecogs.com/gif.latex?%24g_1%20%3D%20e_%7Bmother%7D-e_%7Bfather%7D%24%2C%5C%20%24g_2%20%3D%20e_%7Bgirl%7D-e_%7Bboy%7D%24), etc. and average over them. But just using ![eq](http://latex.codecogs.com/gif.latex?%24e_%7Bwoman%7D-e_%7Bman%7D%24) will give good enough results.)

### Names and their gender bias
List of names and their similarities with constructed vector:
john -0.23163356145973724      
marie 0.315597935396073      
sophie 0.31868789859418784      
ronaldo -0.31244796850329437      
priya 0.176320418390094      
rahul -0.16915471039231722      
danielle 0.24393299216283895      
reza -0.07930429672199552      
katy 0.2831068659572615      
yasmin 0.23313857767928758      

As you can see, female first names tend to have a positive cosine similarity with our constructed vector ![eq](http://latex.codecogs.com/gif.latex?%24g%24), while male first names tend to have a negative cosine similarity. This is not surprising, and the result seems acceptable.

### Other words and their similarities:
lipstick 0.2769191625638267     
guns -0.1888485567898898     
science -0.060829065409296994     
arts 0.008189312385880328     
literature 0.06472504433459927     
warrior -0.2092016464112529     
doctor 0.11895289410935041     
tree -0.07089399175478091     
receptionist 0.33077941750593737     
technology -0.131937324475543     
fashion 0.03563894625772699     
teacher 0.17920923431825664     
engineer -0.08039280494524072     
pilot 0.001076449899191679     
computer -0.10330358873850498     
singer 0.1850051813649629     

It is astonishing how these results reflect certain unhealthy gender stereotypes. For example, "computer" is closer to "man" while "literature" is closer to "woman". Ouch!

Note that some word pairs such as "actor"/"actress" or "grandmother"/"grandfather" should remain gender specific, while other words such as "receptionist" or "technology" should be neutralized, i.e. not be gender-related. You will have to treat these two type of words differently when debiasing.

### 3.1 - Neutralize bias for non-gender specific words

The figure below should help we visualize what neutralizing does. If we're using a 50-dimensional word embedding, the 50 dimensional space can be split into two parts: The bias-direction ![eq](http://latex.codecogs.com/gif.latex?%24g%24), and the remaining 49 dimensions, which we'll call ![eq](http://latex.codecogs.com/gif.latex?%24g_%7B%5Cperp%7D%24). Here these 49 dimensional ![eq](http://latex.codecogs.com/gif.latex?%24g_%7B%5Cperp%7D%24) is perpendicular (or "othogonal") to ![eq](http://latex.codecogs.com/gif.latex?%24g%24). The neutralization step takes a vector such as ![eq](http://latex.codecogs.com/gif.latex?%24e_%7Breceptionist%7D%24) and zeros out the component in the direction of ![eq](http://latex.codecogs.com/gif.latex?%24g%24), giving us ![eq](http://latex.codecogs.com/gif.latex?%24e_%7Breceptionist%7D%5E%7Bdebiased%7D%24).

Even though ![eq](http://latex.codecogs.com/gif.latex?%24g_%7B%5Cperp%7D%24) is 49 dimensional, given the limitations of what can be drawn on a screen, the illustration is done using a 1 dimensional axis below.

<img src="https://raw.githubusercontent.com/00arun00/DeBiasingWE/master/images/neutral.png" style="width:800px;height:300px;">
<caption><center> **Figure 2**: The word vector for "receptionist" represented before and after applying the neutralize operation. </center></caption>

Given an input embedding $e$, we can use the following formulas to compute ![eq](http://latex.codecogs.com/gif.latex?%24e%5E%7Bdebiased%7D%24):

![eq](http://latex.codecogs.com/gif.latex?%24%24e%5E%7Bbias%5C_component%7D%20%3D%20%5Cfrac%7Be%20%5Ccdot%20g%7D%7B%7C%7Cg%7C%7C_2%5E2%7D%20*%20g%24%24)   
![eq](http://latex.codecogs.com/gif.latex?%24%24e%5E%7Bdebiased%7D%20%3D%20e%20-%20e%5E%7Bbias%5C_component%7D%24%24)

$e^{bias\_component}$ is the projection of $e$ onto the direction ![eq](http://latex.codecogs.com/gif.latex?%24g%24).

### 3.2 - Equalization algorithm for gender-specific words

Next, lets see how debiasing can also be applied to word pairs such as "actress" and "actor." Equalization is applied to pairs of words that we might want to have differ only through the gender property. As a concrete example, suppose that "actress" is closer to "babysit" than "actor." By applying neutralizing to "babysit" we can reduce the gender-stereotype associated with babysitting. But this still does not guarantee that "actor" and "actress" are equidistant from "babysit." The equalization algorithm takes care of this.

The key idea behind equalization is to make sure that a particular pair of words are equi-distant from the 49-dimensional ![eq](http://latex.codecogs.com/gif.latex?%24g_%5Cperp%24). The equalization step also ensures that the two equalized steps are now the same distance from ![eq](http://latex.codecogs.com/gif.latex?%24e_%7Breceptionist%7D%5E%7Bdebiased%7D%24), or from any other work that has been neutralized. In pictures, this is how equalization works:

<img src="https://raw.githubusercontent.com/00arun00/DeBiasingWE/master/images/equalize10.png" style="width:800px;height:400px;">


The derivation of the linear algebra to do this is a bit more complex. (See Bolukbasi et al., 2016 for details.) But the key equations are:

![eq](http://latex.codecogs.com/gif.latex?%24%24%20%5Cmu%20%3D%20%5Cfrac%7Be_%7Bw1%7D%20&plus;%20e_%7Bw2%7D%7D%7B2%7D%24%24)    

![eq](http://latex.codecogs.com/gif.latex?%24%24%20%5Cmu_%7BB%7D%20%3D%20%5Cfrac%20%7B%5Cmu%20%5Ccdot%20bias%5C_axis%7D%7B%7C%7Cbias%5C_axis%7C%7C_2%5E2%7D%20*bias%5C_axis%20%24%24)       

![eq](http://latex.codecogs.com/gif.latex?%24%24%5Cmu_%7B%5Cperp%7D%20%3D%20%5Cmu%20-%20%5Cmu_%7BB%7D%20%24%24)        

![eq](http://latex.codecogs.com/gif.latex?%24%24%20e_%7Bw1B%7D%20%3D%20%5Cfrac%20%7Be_%7Bw1%7D%20%5Ccdot%20bias%5C_axis%7D%7B%7C%7Cbias%5C_axis%7C%7C_2%5E2%7D%20*bias%5C_axis%20%24%24)     
![eq](http://latex.codecogs.com/gif.latex?%24%24%20e_%7Bw2B%7D%20%3D%20%5Cfrac%20%7Be_%7Bw2%7D%20%5Ccdot%20bias%5C_axis%7D%7B%7C%7Cbias%5C_axis%7C%7C_2%5E2%7D%20*bias%5C_axis%20%24%24)      


![eq](http://latex.codecogs.com/gif.latex?%24%24e_%7Bw1B%7D%5E%7Bcorrected%7D%20%3D%20%5Csqrt%7B%20%7C%7B1%20-%20%7C%7C%5Cmu_%7B%5Cperp%7D%20%7C%7C%5E2_2%7D%20%7C%7D%20*%20%5Cfrac%7Be_%7B%5Ctext%7Bw1B%7D%7D%20-%20%5Cmu_B%7D%20%7B%7C%28e_%7Bw1%7D%20-%20%5Cmu_%7B%5Cperp%7D%29%20-%20%5Cmu_B%29%7C%7D%24%24)     

![eq](http://latex.codecogs.com/gif.latex?%24%24e_%7Bw2B%7D%5E%7Bcorrected%7D%20%3D%20%5Csqrt%7B%20%7C%7B1%20-%20%7C%7C%5Cmu_%7B%5Cperp%7D%20%7C%7C%5E2_2%7D%20%7C%7D%20*%20%5Cfrac%7Be_%7B%5Ctext%7Bw2B%7D%7D%20-%20%5Cmu_B%7D%20%7B%7C%28e_%7Bw2%7D%20-%20%5Cmu_%7B%5Cperp%7D%29%20-%20%5Cmu_B%29%7C%7D%24%24)     
![eq](http://latex.codecogs.com/gif.latex?%24%24e_1%20%3D%20e_%7Bw1B%7D%5E%7Bcorrected%7D%20&plus;%20%5Cmu_%7B%5Cperp%7D%24%24)     
![eq](http://latex.codecogs.com/gif.latex?%24%24e_2%20%3D%20e_%7Bw2B%7D%5E%7Bcorrected%7D%20&plus;%20%5Cmu_%7B%5Cperp%7D%24%24)   


## Results
```
cosine similarities before equalizing:      
cosine_similarity(word_to_vec_map["man"], gender) =  -0.1171109576533683      
cosine_similarity(word_to_vec_map["woman"], gender) =  0.3566661884627037      

cosine similarities after equalizing:      
cosine_similarity(e1, gender) =  -0.7165727525843935      
cosine_similarity(e2, gender) =  0.7396596474928909      
```

**References**:
- The debiasing algorithm is from Bolukbasi et al., 2016, [Man is to Computer Programmer as Woman is to
Homemaker? Debiasing Word Embeddings](https://papers.nips.cc/paper/6228-man-is-to-computer-programmer-as-woman-is-to-homemaker-debiasing-word-embeddings.pdf)
- The GloVe word embeddings were due to Jeffrey Pennington, Richard Socher, and Christopher D. Manning. (https://nlp.stanford.edu/projects/glove/)
