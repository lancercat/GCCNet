import smtplib
from email.message import EmailMessage
import os;
from utils.libpath import pathcfg;
class nekoexpnotification:
    UNAME="nekoexpnotification@vivaldi.net";
    PWD="neko.pub.notification";
    HOST="smtp.vivaldi.net"
    PORT=587;

    def send_msg(this,dst,content,subj="[ER] The experiment result"):
        # try :
        #     msg = EmailMessage()
        #     msg.set_content(content);
        #     msg['Subject'] = subj;
        #     msg['From'] = this.UNAME;
        #     msg['To']=dst;
        #     smtp = smtplib.SMTP(this.HOST,this.PORT)
        #     smtp.ehlo();
        #     smtp.starttls();
        #     smtp.login(this.UNAME, this.PWD);
        #     smtp.sendmail(this.UNAME, dst, msg.as_string());
        #     smtp.close();
        # except:
            fp=open(os.path.join(pathcfg.generic_data_root,"notifier.lost"),"a+");
            fp.write(content);
            fp.write("\n------------------------------------\n");
            fp.close();


