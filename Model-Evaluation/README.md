# Model Evaluation 〽️

เทคนิคการประเมินผลของโมเดล เป็นสิ่งสำคัญในการประเมินประสิทธิภาพและความเหมาะสมของโมเดล machine learning ช่วยในการเข้าใจถึงประสิทธิภาพในการมาประยุกต์ใช้งานจริงใน model และเพื่อทราบความเหมาะสมในการนำไปใช้งาน

## แบ่งออกเป็น 3 เทคนิค

1. KFold Cross-Validation 
- เป็นเทคนิคที่ใช้สำหรับประเมินความสามารถในการทำนายของโมเดลโดยการแบ่งข้อมูลออกเป็น K ชุด (folds) และใช้แต่ละ fold เป็นชุดทดสอบในขณะที่ฝึกโมเดลด้วย fold ที่เหลือ

2. Reseb Validation 
- เป็นเทคนิคที่โมเดลจะถูกฝึกด้วยข้อมูลทั้งหมดที่มีอยู่นอกจากชุด validation ขนาดเล็ก (reseb) ซึ่งใช้สำหรับประเมินประสิทธิภาพของโมเดล

3. Hold-out Validation 
- เป็นเทคนิคที่ใช้ง่ายๆ โดยการเก็บข้อมูลบางส่วนเป็นชุดทดสอบในขณะที่ส่วนที่เหลือใช้สำหรับการฝึก วิธีนี้ง่ายแต่อาจทำให้มีความผันผวนในผลลัพธ์ตามการเลือกสุ่ม

## Assignments

### [KFold](https://github.com/MLol-3/Machine-learning-class67/blob/b37a33e434a4bc87d03d3c083c3ef68f6618a8bc/Model-Evaluation/kfold.ipynb)


### [Resub](https://github.com/MLol-3/Machine-learning-class67/blob/b37a33e434a4bc87d03d3c083c3ef68f6618a8bc/Model-Evaluation/Resub.ipynb)


### [Hold out]()
