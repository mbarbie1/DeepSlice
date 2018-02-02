# -*- coding: utf-8 -*-
"""
TODO this is still work in progress!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

Created on Sat Jan  6 11:56:44 2018

@author: mbarbier
"""

# Import smtplib for the actual sending function
import smtplib
# Import the email modules we'll need
from email.mime.text import MIMEText
from datetime import datetime

def sendMessage( textFilePath, subject, from_address, to_address ):
    # Open a plain text file for reading.  For this example, assume that
    # the text file contains only ASCII characters.
    with open( textFilePath ) as fp:
        # Create a text/plain message
        msg = MIMEText(fp.read())
    
    # me == the sender's email address
    # you == the recipient's email address
    msg['Subject'] = subject
    msg['From'] = from_address
    msg['To'] = to_address
    
    # Send the message via our own SMTP server.
    #s = smtplib.SMTP('localhost')

    msg = 'Hello world.'

    server = smtplib.SMTP('smtp.gmail.com',587) #port 465 or 587
    server.ehlo()
    server.starttls()
    server.ehlo()
    server.login('michael.barbier@gmail.com','ui,.kom,.')
    server.sendmail(from_address,to_address,msg)
    server.close()

    #s = smtplib.SMTP(host='smtp-mail.outlook.com', port=587)
    #s.send_message(msg)
    #s.quit()

textFilePath = "/home/mbarbier/Documents/prog/DeepSlice/python/keras_small/appdata/run_finish_mail.txt"
subject = "Run finished: " + str(datetime.now())
to_address = "michael.barbier@uantwerpen.be"
from_address = "michael.barbier@gmail.com"
sendMessage( textFilePath, subject, from_address, to_address )
