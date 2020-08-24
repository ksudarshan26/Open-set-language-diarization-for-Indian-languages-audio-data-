#Still on devoloping

from tkinter import *
from tkinter import filedialog
import tkinter as tk
def raise_frame(frame):
    frame.tkraise()
root=Tk()
root.title("Automatic Language Segmentation and Identification")
ic=PhotoImage(file="./images/icon.png")
root.iconphoto(False , ic)
def quit1():
    root.destroy()
def hello1(fz):
    print(fz)
def Load():
            fz=tk.filedialog.askopenfilename(initialdir="/",title="Open File",filetypes=(("WAV Files", "*.wav"), ("All Files", "*.*")))
            hello1(fz)
home=Frame(root,bg="darkblue")
firstpage=Frame(root,bg="cyan2")
secondpage=Frame(root,bg="darkblue")
for frame in (home,firstpage,secondpage):
    frame.grid(row=0,column=0,sticky='news')

main_img=PhotoImage(file="./images/main.png")
first_img=PhotoImage(file="./images/main2.png")
bt1_img=PhotoImage(file="./images/cnt.png")
bt_back=PhotoImage(file="./images/back.png")
bt_quit=PhotoImage(file="./images/quit.png")
bt_open=PhotoImage(file="./images/load.png")
bt_mic=PhotoImage(file="./images/speak.png")

Button(home,image=main_img,command=lambda:raise_frame(firstpage)).pack()

label1=Label(firstpage,image=first_img).place(x=0,y=0,relwidth=1,relheight=1)
Button(firstpage,image=bt1_img,borderwidth=0,command=lambda:raise_frame(secondpage)).pack(side='right',anchor='se',padx=60,pady=5)
Button(firstpage,image=bt_back,border=0,command=lambda:raise_frame(home)).pack(side='left',anchor='sw')
Button(firstpage,image=bt_mic,borderwidth=0,bd=0).place(x=250,y=350)
Button(firstpage,image=bt_open,borderwidth=0,bd=0,command=Load).place(x=1000,y=350)
Button(firstpage,image=bt_quit,borderwidth=0,command=quit1).place(x=750,y=798)

#label2=Label(secondpage,image=first_img).place(x=0,y=0,relwidth=1,relheight=1)
label3=Label(secondpage,text="Madanapalle Institute Of Technology and Science,Madanapalle\n\nContact us:\n\nVinay Kumar G N\nE-Mail: 18691a4l5@mits.ac.in\n\nVarsha Reddy P\nE-Mail: 18695a04k8@mits.ac.in\n\nSudarshan K\nE-Mail: 18691a04i5@mits.ac.in\n\nSangeetha L\nE-Mail: 18691a0586@mits.ac.in\n\nVenkata Harish B\nE-Mail: 18691a04l0@mits.ac.in\n\nSukanya P\nE-Mail: 18691a04j0@mits.ac.in",font=("Lucida Bright",24,"bold"),fg="white",bg="darkblue").pack(pady=40)
Button(secondpage,image=bt_back,borderwidth=0,command=lambda:raise_frame(firstpage)).place(x=0,y=798)
Button(secondpage,image=bt_quit,borderwidth=0,command=quit1).place(x=750,y=798)
Label(secondpage,text="-HunTer_SQuaD",font=("Lucida Bright",20,"bold"),bg="darkblue",fg="white").place(x=1300,y=798)



raise_frame(home)
root.mainloop()
