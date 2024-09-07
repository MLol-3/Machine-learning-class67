# Machine-Learning-ClassML📈

สำหรับการรวมและส่งงานในรายวิชา การเรียนรู้ของเครื่อง ปีการศึกษา 2567

## แบบฝึกหัดเขียนโปรแกรมครั้งที่ 1 Linear Regression
- ### [Linear Regression](https://github.com/MLol-3/Machine-learning-class67/tree/b37a33e434a4bc87d03d3c083c3ef68f6618a8bc/Linear%20Regression)
1. เขียนโปรแกรมสำหรับสร้างแบบจำลองเชิงเส้นด้วยวิธีลดตามความชัน พร้อมทั้งแสดงฟังก์ชันค่าใช้จ่ายในรูปของคอนทัวร์และแสดงให้เห็นถึงขั้นตอนในการปรับพารามิเตอร์ (Lecture หน้าที่ 49)

2. เขียนโปรแกรมสำหรับแสดงผลกระทบต่อการทำงานของวิธีลดตามความชันและฟังก์ชันค่าใช้จ่าย เมื่อตัวแปร x หลายตัวมีค่าแตกต่างกันมาก และแสดงผลของการปรับปรุงประสิทธิภาพด้วยการทำให้เป็นมาตรฐาน (Lecture หน้าที่ 61)

3. เขียนโปรแกรมสำหรับแสดงผลของการปรับพารามิเตอร์การเรียนรู้ (Lecture หน้าที่ 66)

4. เขียนโปรแกรมสำหรับเปรียบเทียบผลลัพธ์ที่ได้จากวิธีสมการปรกติและวิธีลดตามความชัน (Lecture หน้าที่ 57 และ 59)

## แบบฝึกหัดเขียนโปรแกรมครั้งที่ 2 Generalization and Model Evaluation
- ### [Generalization](https://github.com/MLol-3/Machine-learning-class67/tree/b37a33e434a4bc87d03d3c083c3ef68f6618a8bc/Generalization)
- ### [Model Evaluation](https://github.com/MLol-3/Machine-learning-class67/tree/b37a33e434a4bc87d03d3c083c3ef68f6618a8bc/Model-Evaluation)
1. เขียนโปรแกรมเพื่อทดสอบความเที่ยงตรง (Precision) และการทดสอบความแม่นยํา (Accuracy) ของวิธี Resubstitution, Holdout และ Cross Validation โดยใช้ข้อมูล height weight โดยให้ออกแบบการทดลองเอง 


2. หาค่าความเอนเอียงและความแปรปรวนด้วย analytical method และ simulation ของ 1.แบบจำลองค่าคงที่ 2.แบบจำลองเชิงเส้นและ 3.แบบจำลองเชิงเส้นผ่านจุดกำเนิด

    2.1 เมื่อกำหนดให้ฟังก์ชันเป้าหมายคือ $sin(\pi x)$ และสุ่มข้อมูลด้วยการแจกแจงแบบเอกรูปออกมา 2 ตัวอย่างในช่วง [-1,1] (Lecture หน้า 18)

    2.2 เมื่อกำหนดให้ฟังก์ชันเป้าหมายคือ $x^2$ และสุ่มข้อมูลด้วยการแจกแจงแบบเอกรูปออกมา 2 ตัวอย่างในช่วง [-1,1] 

3. เขียนโปรแกรมสำหรับเส้นโค้งการเรียนรู้เปรียบเทียบระหว่างแบบจำลองค่าคงที่ แบบจำลองเชิงเส้น แบบจำลองเชิงเส้นผ่านจุดกำเนิด และทดลองเพิ่มเติมด้วยการใส่สัญญาณรบกวน (Lecture หน้า 35-37)

ป.ล. ให้ใช้ normal equation หรือ lib ในการสร้างแบบจำลอง ไม่แนะนำให้ใช้ gradient descent
https://en.wikipedia.org/wiki/Expected_value

## แบบฝึกหัดเขียนโปรแกรมครั้งที่ 3 Model Selection
- ### [Model Selection](https://github.com/MLol-3/Machine-learning-class67/tree/28ae5f71d14d98e125390c7d956ea2edd2da20c1/Model%20Selection)
1. ทดลองซ้ำและการปรับพารามิเตอร์ให้มากกว่าใน lecture หรือลอง generate date ที่แตกต่างจากที่เรียน

2. เขียนโปรแกรมสำหรับการทำ Nested Cross-Validation และออกแบบการทดลองเพื่อแสดงให้เห็นถึงความจำเป็นของการทำสองลูปแทนที่จะทำเพียงแค่ลูปเดียว

## แบบฝึกหัดเขียนโปรแกรมครั้งที่ 5 Linear Model Selection and Regularization
- ### [Linear Model Selection]()
- ### [Regularizationn]()
1. เขียนโปรแกรมแสดงความสัมพันธ์ระหว่าง model และ cost function ของ ridge regression (Lecture หน้าที่ 37)

2. เขียนโปรแกรมแสดงความสัมพันธ์ระหว่าง complexity ของ model และ etrain, etest ของ ridge regression (Lecture หน้าที่ 44)

3. เขียนโปรแกรมสำหรับเปรียบเทียบค่าคลาดเคลื่อนนอกตัวอย่างของแบบจำลองเชิงเส้นที่มีและไม่มีการทำให้เป็นปรกติ (Lecture หน้าที่ 48-49)

## แบบฝึกหัดเขียนโปรแกรมครั้งที่ 6 logistic regression
1. เขียนโปรแกรมสำหรับสร้างแบบจำลอง logistic regression ด้วยวิธี gradient descent สำหรับแก้ปัญหา logic AND, OR และ XOR (สำหรับปัญหา XOR ต้องใช้ interaction feature)

2. ใช้โปรแกรมที่ได้จากข้อ 1 แก้ปัญหาจาก data ที่ generate ขึ้นมา โดยต้องอธิบายด้วยว่า data แต่ละชุด generate ขึ้นมาได้อย่างไร ตัวอย่างข้อมูล เช่น https://scikit-learn.org/0.15/auto_examples/plot_classifier_comparison.html

- ทั้งสองข้อให้ plot decision boundary ของ logistic regression ออกมาแสดงผลด้วย

3. ใช้ logistic regression แก้ปัญหา handwritten digit recognition จากชุดข้อมูล MNIST โดยทำ 2 แบบจำลอง คือ binary classification และ multi-class classification โดย binary classification ให้เลือกข้อมูลออกมาสอง class เช่น classify รูปเลข 0 และ 1 เป็นต้น ขณะที่ multi-class classification ให้ classify เลข 0-9 ออกจากกันให้ได้ ทั้งสองแบบจำลองให้นำ weight มา plot เป็นรูปเพื่อแสดงความเข้าใจการทำงานของ weight ด้วย

## แบบฝึกหัดเขียนโปรแกรมครั้งที่ 6 Bayes Decision Theory
1. เขียนโปรแกรมสำหรับสร้างตัวจำแนกแบบเบส์สำหรับการแจกแจงปรกติตัวแปรเดียว กรณีที่ความแปรปรวนของทั้งสองคลาสเท่ากัน

2. เขียนโปรแกรมสำหรับสร้างตัวจำแนกแบบเบส์สำหรับการแจกแจงปรกติตัวแปรเดียว กรณีที่ความแปรปรวนของทั้งสองคลาสไม่เท่ากัน

3. เขียนโปรแกรมสำหรับสร้างตัวจำแนกกำลังสอง

4. เขียนโปรแกรมสำหรับสร้างตัวจำแนกเชิงเส้น
  #### (ทั้ง 4 ข้อให้ วาดกราฟ likelihood, posterior และขอบตัดสินใจโดยทำสองรูปแบบ คือ)
- กำหนดค่าพารามิเตอร์ของการแจกแจก 
- สุ่มตัวอย่างเพื่อนำมาคำนวณค่าพารามิเตอร์ของการแจกแจง 


5. เขียนโปรแกรมสำหรับ plot decision boundary เปรียบเทียบระหว่าง LDA, QDA และ Logistic regression 
- โดยอาจจะมีการเพิ่มพจน์ second order polynomial โดยอาจจะใช้การสุ่มข้อมูลในรูปแบบต่างๆดังนี้ https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html#sphx-glr-auto-examples-classification-plot-classifier-comparison-py หรือ https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=circle®Dataset=reg-plane&learningRate=0.03®ularizationRate=0&noise=0&networkShape=4,2&seed=0.87693&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&
