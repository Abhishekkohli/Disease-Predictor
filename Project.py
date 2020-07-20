from tkinter import *
import Project_backend as pr 
import pandas as pd
gui=Tk()
gui.geometry("800x750")

gui.title("Disease Predictor")

list_variables=[]
for i in range(len(pr.features3)):
    list_variables.append(IntVar())

list_symptoms=[]
for i in range(len(pr.features3)):
    chkbn=Checkbutton(gui,text=pr.features3[i],variable=list_variables[i],onvalue=1,offvalue=0)
    list_symptoms.append(chkbn)
a=StringVar()
b=StringVar()
c=StringVar()
d=StringVar()

l1=Label(gui,text="Support Vector Machine:" , bg = "cyan", bd=4)
l11=Label(gui,textvariable=a , bg = "red")
l2=Label(gui,text="Logistic Regression:" , bg = "cyan", bd=4)
l22=Label(gui,textvariable=b ,bg="red")
l3=Label(gui,text="K nearest Neighbors:" , bg = "cyan", bd=4)
l33=Label(gui,textvariable=c ,bg="red")
l4=Label(gui,text="Random Forest Classifier:" , bg = "cyan", bd=4)
l44=Label(gui,textvariable=d ,bg="red")

def predict_disease() :
    df_row=[]
    for i in list_variables:
        df_row.append(i.get())
    df=[]
    df.append(df_row)
    predict_df=pd.DataFrame(df,columns=pr.features3)
    num1=pr.model_list[0].predict(predict_df)
    num2=pr.model_list[1].predict(predict_df)
    num3=pr.model_list[2].predict(predict_df)
    num4=pr.model_list[3].predict(predict_df)
    for val in pr.label_to_cat:
        if val[0] == num1:
            a.set( val[1] + "(Accuracy:" + str(pr.accuracy_models[0]*100) + "%")
            l11.config(textvariable=a)
        if val[0] == num2:
            b.set(val[1] + "(Accuracy:" + str(pr.accuracy_models[1]*100) + "%")
            l22.config(textvariable=b)
        if val[0] == num3:
            c.set(val[1] + "(Accuracy:" + str(pr.accuracy_models[2]*100) + "%")
            l33.config(textvariable=c)
        if val[0] == num4:
            d.set(val[1] + "(Accuracy:" + str(pr.accuracy_models[3]*100) + "%")
            l44.config(textvariable=d)

button=Button(gui,text="predict",activebackground="red",activeforeground="yellow",fg="black",command=predict_disease)

button.pack(fill=X,expand=True)
sum_x=30
sum_y=20
for i in range(len(pr.features3)):
    list_symptoms[i].place(x=sum_x,y=sum_y)
    sum_y+=22
    if sum_y > 400 :
        sum_y=20
        sum_x+=200

button.place(x=290,y=500)

l1.place(x=30,y=550)
l11.place(x=180,y=550)
l2.place(x=450,y=550)
l22.place(x=600,y=550)
l3.place(x=30,y=620)
l33.place(x=180,y=620)
l4.place(x=450,y=620)
l44.place(x=600,y=620)
gui.mainloop()







