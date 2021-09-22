import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

dataset=pd.read_csv("collegePlace.csv")
#print(dataset)


#To count % of IT students placed:
count_i=0
count_p=0
for i in dataset["Stream"]:
    if i=="Information Technology":
        count_i+=1
#print(count_i)
for i in range(0,len(dataset.index)):
    if dataset.iloc[i,2]=="Information Technology" and dataset.iloc[i,7]==1:
        count_p+=1
#print(count_p)
percent_p=(count_p/count_i)*100
print("\nPercentage of Information Technology students placed = ",percent_p)


#To plot graph of CGPA vs PlacedOrNot
x=dataset['CGPA'].values
#print(x)
y=dataset.PlacedOrNot.values
#print(y)
plt.style.use('dark_background')
ax=plt.axes()
ax.set_facecolor("cyan")
plt.scatter(x,y,c='r')
plt.title("CGPA vs PlacedOrNot")
plt.xlabel("CGPA")
plt.ylabel("Placed or Not")
plt.show()


#To predict in a student will get placed or not
print("\n\t\t\tPrediction of Placement:")
feature_cols=['Age','Internships','CGPA','Hostel','HistoryOfBacklogs']
x=dataset[feature_cols]
#print(x)
y=dataset.PlacedOrNot
#print(y)
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)
from sklearn.linear_model import LogisticRegression
logreg=LogisticRegression()
logreg.fit(x_train,y_train)
y_pred=logreg.predict(x_test)
#stud_info=[22,1,8,1,1]
stud_info=[]
age=int(input("Enter Age: "))
stud_info.append(age)
intern=int(input("Enter no. of Internships: "))
stud_info.append(intern)
cgpa=int(input("Enter CGPA: "))
stud_info.append(cgpa)
hostel=int(input("Student lives in Hostel or not (1/0)? "))
stud_info.append(hostel)
back=int(input("Is there any History of Backlog (1/0)? "))
stud_info.append(back)
y_pred1=logreg.predict([stud_info])
print("\nGiven Info about Student: ",stud_info)
#print(y_pred1)
if y_pred1==[1]:
    print("Great! The Student will probably be placed.")
elif y_pred1==[0]:
    print("Sorry, there are very less chances of the student getting placed :(")


