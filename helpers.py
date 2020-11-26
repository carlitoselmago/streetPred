from datetime import datetime, timedelta

class helpers():
    def __init__(self):
        pass

    def timeBlockIndex(self,splitblocks,start,end):
        date=datetime.fromtimestamp((start+((end-start)/2))/1000.0).replace(hour=0, minute=0, second=0, microsecond=0)
        daybeginMS=int(date.strftime("%s")) * 1000

        end=86400000 #a day in milis
        block=end/splitblocks
        duration=(end-daybeginMS)-(end-start)
        needleE=(duration)/2


        for b in range(splitblocks):
            rStart=b*block
            rEnd=(b+1)*block

            if needleE>rStart and needleE<rEnd:

                return b

        return splitblocks-1
