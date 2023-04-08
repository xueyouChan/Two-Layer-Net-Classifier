# TwoLayerNet Classifier

代码总共有两部分，一个是MNIST数据集的导入，是mnist.py，另一个是主代码文件Model.py。

<p>代码的运行主要是Model.py文件，在运行前需要注意这几个部分要把路径设置为自己的路径</p>
<pre><code>sys.path.append('Add Your Path')  # 代码第六段
net.save_model("Add Your Path") # train_test函数最后一行
net.load_model('Add Your Path') # __main__部分
</code></pre>

关于参数查找：</br>
  &nbsp;&nbsp;&nbsp;&nbsp; 学习率的候选是[0.01, 0.005, 0.001];</br>
  &nbsp;&nbsp;&nbsp;&nbsp; 隐藏层大小的候选是[100, 200, 300];</br>
  &nbsp;&nbsp;&nbsp;&nbsp; 正则化强度的候选是[0.01, 0.005, 0.001]。</br>
最好的参数组合是学习率为0.005，隐藏层大小为300，正则化强度为0.001，分类准确率为97.89%。

关于最后的可视化，得到的结果如下：</br>
<center>
  ![Plot]("C:/Users/Lenovo/研一下/深度学习/plot.jpg")
</center>
