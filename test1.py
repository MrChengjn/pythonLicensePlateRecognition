import os
import tkinter as tk
from tkinter import *
from tkinter import ttk
import tkinter.font as tkFont
from PIL import Image, ImageTk
from tkinter import filedialog
from main import MainRecognition

class StartPage:
	def __init__(self, parent_window):
		parent_window.destroy() # 销毁子界面

		self.window = tk.Tk()  # 初始框的声明
		self.window.title('车牌识别系统')
		self.window.geometry('300x410')

		label = Label(self.window, text="车牌识别系统", font=("Verdana", 20))
		label.pack(pady=100)  # pady=100 界面的长度

		Button(self.window, text="进入识别", font=tkFont.Font(size=16), command=lambda: discern(self.window), width=30, 
               height=2,fg='white', bg='gray', activebackground='black', activeforeground='white').pack()
		Button(self.window, text="识别说明", font=tkFont.Font(size=16), command=lambda: AdminPage(self.window), width=30,
			   height=2,fg='white', bg='gray', activebackground='black', activeforeground='white').pack()
		Button(self.window, text='退出系统', height=2, font=tkFont.Font(size=16), width=30, command=self.window.destroy,
			   fg='white', bg='gray', activebackground='black', activeforeground='white').pack()

		self.window.mainloop() # 主消息循环


class discern:
    def __init__(self, parent_window):
        parent_window.destroy() # 销毁主界面
        self.window = Tk()  # 初始框的声明
        self.window.title('车牌识别界面')
        self.recognition=MainRecognition()
        
        style_head = ttk.Style()
        style_head.configure("Treeview.Heading", font=("黑体", 14))
        
        self.center = tk.Frame(width=700, height=600)
        self.cols = ("图片名称", "识别颜色", "识别牌号", "车牌信息","识别用时")
        self.tree = ttk.Treeview(self.center, show="headings", height=30, columns=self.cols)
        self.vbar = ttk.Scrollbar(self.center, orient=VERTICAL, command=self.tree.yview)
		# 定义树形结构与滚动条
        self.tree.configure(yscrollcommand=self.vbar.set)
        
        self.t_list= []
        
        # 表格的标题
        self.tree.column("图片名称", width=150, anchor='center')
        self.tree.column("识别颜色", width=150, anchor='center')
        self.tree.column("识别牌号", width=150, anchor='center')
        self.tree.column("车牌信息", width=150, anchor='center')
        self.tree.column("识别用时", width=100, anchor='center')
        
        # 调用方法获取表格内容插入
        self.tree.grid(row=0, column=0, sticky=NSEW)
        self.vbar.grid(row=0, column=1, sticky=NS)
        
        for i in range(len(self.t_list)):  # 写入数据
            self.tree.insert('', i, values=(self.t_list[i][0],self.t_list[i][1],self.t_list[i][2],self.t_list[i][3],self.t_list[i][4]))
        for col in self.cols:  # 绑定函数，使表头可排序
            self.tree.heading(col, text=col,command=lambda _col=col: self.tree_sort_column(self.tree, _col, False))
        
        self.center.grid(row=0, column=0, padx=5, pady=5)
        self.center.grid_propagate(0)
        self.center.tkraise() # 开始显示主菜单
        
        self.photo = tk.Frame(width=600, height=600)
        
        
        self.button1 = ttk.Button(self.photo, text='导入图片', width=20, command=self.insert)
        self.button1.grid(row=0, column=0, columnspan=1,sticky=W)
        
        
        self.photo.grid(row=0, column=1, padx=5, pady=5)
        self.photo.grid_propagate(0)
        self.photo.tkraise() # 开始显示主菜单
        
        
        
        self.window.protocol("WM_DELETE_WINDOW", self.back)  # 捕捉右上角关闭点击
        self.window.mainloop()  # 进入消息循环
        
        
    def tree_sort_column(self, tv, col, reverse):
        l = [(tv.set(k, col), k) for k in tv.get_children('')]
        l.sort(reverse=reverse)  # 排序方式
		# rearrange items in sorted positions
        for index, (val, k) in enumerate(l):  # 根据排序后索引移动
            tv.move(k, '', index)
        tv.heading(col, command=lambda: self.tree_sort_column(tv, col, not reverse))  # 重写标题，使之成为再点倒序的标题
    def back(self):
        StartPage(self.window) # 显示主窗口 销毁本窗口
        
    def insert(self):
        self.file_path = filedialog.askopenfilename()
        self.img = Image.open(self.file_path)
        w,h=self.img.size
        f1 = 1.0*600/w 
        f2 = 1.0*600/h
        factor = min([f1, f2])
        width = int(w*factor)
        height = int(h*factor)
        self.img = self.img.resize((width,height))
        self.img = ImageTk.PhotoImage(self.img)  # 用PIL模块的PhotoImage打开
        self.imglabel = Label(self.photo,image=self.img,height=600,width=600)
        self.imglabel.grid(row=1, column=0, columnspan=1)
        self.photo.grid(row=0, column=1, padx=5, pady=5)
        self.photo.grid_propagate(0)
        self.photo.tkraise() # 开始显示主菜单
        
        list_temp=[]
        name = os.path.basename(self.file_path)
        result = self.recognition.VLPR(self.file_path)
        list_temp.append(name)
        list_temp.append(result['Type'])
        list_temp.append(result['Number'])
        list_temp.append(result['From'])
        list_temp.append(result['UseTime'])
        self.t_list.append(list_temp)
        self.tree.insert('',len(self.t_list)-1, values=
                         (name,result['Type'],result['Number'],
                          result['From'],result['UseTime']))
        
if __name__ == '__main__':
    window = tk.Tk()
    StartPage(window)











