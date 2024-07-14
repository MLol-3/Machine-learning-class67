# Machine-Learning-ClassML
## แบบฝึกหัดเขียนโปรแกรมครั้งที่ 1 Linear Regression
1. เขียนโปรแกรมสำหรับสร้างแบบจำลองเชิงเส้นด้วยวิธีลดตามความชัน พร้อมทั้งแสดงฟังก์ชันค่าใช้จ่ายในรูปของคอนทัวร์และแสดงให้เห็นถึงขั้นตอนในการปรับพารามิเตอร์ (Lecture หน้าที่ 49)

2. เขียนโปรแกรมสำหรับแสดงผลกระทบต่อการทำงานของวิธีลดตามความชันและฟังก์ชันค่าใช้จ่าย เมื่อตัวแปร x หลายตัวมีค่าแตกต่างกันมาก และแสดงผลของการปรับปรุงประสิทธิภาพด้วยการทำให้เป็นมาตรฐาน (Lecture หน้าที่ 61)

3. เขียนโปรแกรมสำหรับแสดงผลของการปรับพารามิเตอร์การเรียนรู้ (Lecture หน้าที่ 66)

4. เขียนโปรแกรมสำหรับเปรียบเทียบผลลัพธ์ที่ได้จากวิธีสมการปรกติและวิธีลดตามความชัน (Lecture หน้าที่ 57 และ 59)

## แบบฝึกหัดเขียนโปรแกรมครั้งที่ 2 Generalization and Model Evaluation
1. เขียนโปรแกรมเพื่อทดสอบความเที่ยงตรง (Precision) และการทดสอบความแม่นยํา (Accuracy) ของวิธี Resubstitution, Holdout และ Cross Validation โดยใช้ข้อมูล height weight โดยให้ออกแบบการทดลองเอง 


2. หาค่าความเอนเอียงและความแปรปรวนด้วย analytical method และ simulation ของ 1.แบบจำลองค่าคงที่ 2.แบบจำลองเชิงเส้นและ 3.แบบจำลองเชิงเส้นผ่านจุดกำเนิด

    2.1 เมื่อกำหนดให้ฟังก์ชันเป้าหมายคือ $sin(\pi x)$ และสุ่มข้อมูลด้วยการแจกแจงแบบเอกรูปออกมา 2 ตัวอย่างในช่วง [-1,1] (Lecture หน้า 18)

    2.2 เมื่อกำหนดให้ฟังก์ชันเป้าหมายคือ $x^2$ และสุ่มข้อมูลด้วยการแจกแจงแบบเอกรูปออกมา 2 ตัวอย่างในช่วง [-1,1] 

3. เขียนโปรแกรมสำหรับเส้นโค้งการเรียนรู้เปรียบเทียบระหว่างแบบจำลองค่าคงที่ แบบจำลองเชิงเส้น แบบจำลองเชิงเส้นผ่านจุดกำเนิด และทดลองเพิ่มเติมด้วยการใส่สัญญาณรบกวน (Lecture หน้า 35-37)

ป.ล. ให้ใช้ normal equation หรือ lib ในการสร้างแบบจำลอง ไม่แนะนำให้ใช้ gradient descent
https://en.wikipedia.org/wiki/Expected_value

## แบบฝึกหัดเขียนโปรแกรมครั้งที่ 3 Model Selection
1. ทดลองซ้ำและการปรับพารามิเตอร์ให้มากกว่าใน lecture หรือลอง generate date ที่แตกต่างจากที่เรียน

2. เขียนโปรแกรมสำหรับการทำ Nested Cross-Validation และออกแบบการทดลองเพื่อแสดงให้เห็นถึงความจำเป็นของการทำสองลูปแทนที่จะทำเพียงแค่ลูปเดียว

