# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import MySQLdb as mysql
import nltk

db1 = mysql.connect("localhost", "root", "", "tttn_literature")
db2 = mysql.connect("localhost", "root", "", "lvtn")

cursor1 = db1.cursor()
cursor2 = db2.cursor()

cursor1.execute("SELECT content FROM abstractarticle")

articles = cursor1.fetchall()

sentences = []

exception = 0
for article in articles:
    #sens= article[0].split('[a-zA-Z]*.')
    try:
        print "----------------------\n"
        print article[0]
        sens = nltk.sent_tokenize(article[0])
        sentences.extend(sens)
    except Exception:
        exception+=1
for sentence in sentences:
    if len(sentence)>100 and sentence[0]>='A' and sentence[0]<='Z':
        sql = "INSERT INTO data (sentence) VALUES ('%s')" % (mysql.escape_string(sentence))
        cursor2.execute(sql)
        
db2.commit()

db2.close()
db1.close()



