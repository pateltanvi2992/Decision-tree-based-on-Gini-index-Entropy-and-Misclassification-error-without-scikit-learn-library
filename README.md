# Decision tree based wine quality prediction using Gini index, Entropy and misclassification error without Scikit learn library 
> Decision trees have been found very effective for classification especially in Data Mining. A particular efficient method for classification is the decision tree. The selection of the attribute used at each node of the tree to split the data is crucial in order to correctly classify objects. A split in a decision tree corresponds to the predictor with the maximum separating power. In other words, the best split does the best job in creating nodes where a single class dominates. There are several methods of calculating the predictor's power to separate data. To predict a wine quality here we use the datasets related to red variants of the Portuguese "Vinho Verde" wine. The dataset describes the amount of various chemicals present in wine and their effect on it's quality. . In this study, we have proposed an enhanced version of distributed decision tree algorithm using Gini index, entropy, and misclassification to perform better in terms of model building time without compromising the accuracy. The most significant thing to note here is that we did not use the Scikit Learn library in this project. If you want to use this code, you may also dynamically load any datasets. Additionally, divide the tree depending on the characteristic with the highest information gain and Gini gain by using gain ratio and Gini gain.
# How to run 
  Dataset URL : https://raw.githubusercontent.com/pateltanvi2992/Decision-tree-based-on-Gini-index-Entropy-and-Misclassification-error-without-scikit-learn-library/main/WineQT.csv 
  
  <b>Example to run the python file : </b> 
  python wine_quality_prediction.py --dataset dataset_path
  
  <b>Outputs : </b>
  <p>tree from python code</p>
  <img src = 'https://raw.githubusercontent.com/pateltanvi2992/Decision-tree-based-on-Gini-index-Entropy-and-Misclassification-error-without-scikit-learn-library/main/Tree.PNG' height = 500 width = 500>
  <p> Heatmap diagram </p>
   <img src = 'https://raw.githubusercontent.com/pateltanvi2992/Decision-tree-based-on-Gini-index-Entropy-and-Misclassification-error-without-scikit-learn-library/main/heatmap_wine_data.png' height = 500 width = 500>
<b>REFERENCES :</b>
<ul>
<li> [1] I. Janszky, M. Ericson, M. Blom, A. Georgiades, J. O. Magnusson, H. Alinagizadeh, and S. Ahnve, “Wine drinking is associated with increased heart rate variability in women with coronary heart disease,” Heart, 91(3), pp.314-318, 2005. </li>
<li> [2] V. Preedy, and M. L. R. Mendez, “Wine Applications with Electronic Noses,” in Electronic Noses and Tongues in Food Science, Cambridge, MA, USA: Academic Press, 2016, pp. 137-151. </li>
<li> [3] https://www.kaggle.com/code/andreicosma/decision-trees-misclassification-index </li>
<li> Wang, Y. and Xia, S.T., 2017, March. Unifying attribute splitting criteria of decision trees by Tsallis entropy. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 2507-2511). IEEE. </li>
<li> [4] Zhang, H., Wang, Z., He, J. and Tong, J., 2021, July. Construction of Wine Quality Prediction Model based on Machine Learning Algorithm. In 2021 5th International Conference on Artificial Intelligence and Virtual Reality (AIVR) (pp. 53-58).</li>
<li> [5] Wang, Y. and Xia, S.T., 2017, March. Unifying attribute splitting criteria of decision trees by Tsallis entropy. In 2017 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP) (pp. 2507-2511). IEEE. </li>
<li> [6]  Wang, Y., Song, C. and Xia, S.T., 2015. Unifying decision trees split criteria using tsallis entropy. arXiv preprint arXiv:1511.08136. </li>
<li> [7] Raileanu, L.E. and Stoffel, K., 2004. Theoretical comparison between the gini index and information gain criteria. Annals of Mathematics and Artificial Intelligence, 41(1), pp.77-93.</li>
<li> [8]  Furman, E., Kye, Y. and Su, J., 2019. Computing the Gini index: A note. Economics Letters, 185, p.108753. </li> 
<li> [9] Lerman, R.I. and Yitzhaki, S., 1984. A note on the calculation and interpretation of the Gini index. Economics Letters, 15(3-4), pp.363-368.</li>
<li> [10] https://www.jinhang.work/tech/decision-tree-in-python/ </li>
</ul>
