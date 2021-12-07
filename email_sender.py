from email.header import Header
from email.mime.multipart import MIMEMultipart
from email.mime.application import MIMEApplication
from email.mime.text import MIMEText
from random import randrange
import pandas as pd
import smtplib

# from_addr = 'asdfg@abc.com'  #发件人邮箱
# to_addr = ['qwert@abc.com','zxcvb@abc.com','lkjhg@abc.com'] # 收件人邮箱
def send_email(password,from_addr,to_addr,title,content):
    smtp_server = 'smtp.exmail.qq.com' # 腾讯服务器地址


	# 创建一个带附件的实例
    message = MIMEMultipart() 
    message['From'] = from_addr #发件人
    message['To'] =Header(",".join(to_addr)) #处理多个收件人信息，list 转字符串
    message['Subject'] = Header(title, 'utf-8').encode() # 邮件标题
	

    msg=content
    message.attach(MIMEText(msg,'plain','utf-8'))

	# MIMEApplication对附件进行封装
    # xlsxpart = MIMEApplication(open(filename, 'rb').read())
    # xlsxpart.add_header('Content-Disposition', 'attachment', filename=filename)
    # message.attach(xlsxpart)

	# 发送邮件
    try:
        smtpObj=smtplib.SMTP_SSL(smtp_server) #SMTP的SSL加密方式，端口要用465
        smtpObj.connect(smtp_server,465) #连接腾讯服务器地址，传入地址和端口参数；腾讯企业邮箱STMP端口号是465
        smtpObj.login(from_addr,password) # 登录邮箱，传入发件人邮箱及独立密码
        smtpObj.sendmail(from_addr,to_addr,message.as_string()) 
        print('success')
        smtpObj.quit()
    
    except smtplib.SMTPException as e:
        print('error',e)

group_id = '940234872'
filepath = 'dataset_20210828/{}.csv'.format(group_id) # 已手动删除群主和管理员
df = pd.read_csv(filepath)
qqmails = df['QQ号'].astype('str') + '@qq.com'

print(f'number of all qqmails: {len(qqmails)}')
print('------------------------------------')

password = '9B53Ytk8fS4VvFtH'
from_addr = 'fourier@mail.r3k.com'
title = 'test email'
content = 'test'

current_i = 0
while current_i < len(qqmails):
    to_num = randrange(10, 20)
    if current_i + to_num > len(qqmails):
        to_num = len(qqmails) - current_i
    to_addr = qqmails.iloc[current_i:current_i+to_num].tolist()
    print(f"{current_i}-{current_i+to_num}: {to_addr}")
    print('------------------------------------')

    # to_addr = ['zhuolinlin@sense-sec.com', '838311583@qq.com']
    # send_email(password,from_addr,to_addr,title,content)

    exit(0)

    # 群发间隔时间
    # time.sleep(to_num*10)

    current_i = current_i+to_num






