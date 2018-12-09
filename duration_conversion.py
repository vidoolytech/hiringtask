#Code to convert duration of video into seconds.
j = sl['duration']
time1=[]
a=2
for i in j:
    print("for "+str(a))
    a=a+1
    hour ="0"
    minu="0"
    sec="0"
    mn =i.split("H")
    if len(mn)==2:
        hour = mn[0]
        print(hour)
        ln = mn[1].split("M")
        if len(ln)==2:
            minu= ln[0]
            print(minu)
            kr= ln[1].split("S")
            if len(kr)==2:
                sec = kr[0]
                print(sec)
            if len(kr)==1:
                print(sec)
        if len(ln)==1:
            print(minu)
            kr=ln[0].split("S")
            if len(kr)==2:
                sec = kr[0]
                print(sec)
            if len(kr)==1:
                print(sec)
    if len(mn)==1:
        ln = mn[0].split("M")
        print(hour)
        if len(ln)==2:
            minu= ln[0]
            print(minu)
            kr= ln[1].split("S")
            if len(kr)==2:
                sec = kr[0]
                print(sec)
            if len(kr)==1:
                print(sec)
        if len(ln)==1:
            print(minu)
            kr=ln[0].split("S")
            sec = kr[0]
            print(sec)
            if len(kr)==2:
                sec = kr[0]
                print(sec)
            if len(kr)==1:
                print(sec)
    t = float(hour)*3600+float(minu)*60+float(sec)
    time1.append(t)