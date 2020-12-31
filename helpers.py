from datetime import datetime, timedelta
import re
import xml

class helpers():
    def __init__(self):
        pass

    def timeBlockIndex(self,splitblocks,start,end):
        date=datetime.fromtimestamp((start+((end-start)/2))/1000.0).replace(hour=0, minute=0, second=0, microsecond=0)
        daybeginMS=int(date.strftime("%s")) * 1000

        end=86400000 #a day in milis
        block=end/splitblocks
        duration=(end-daybeginMS)-(end-start)
        needleE=start-daybeginMS#(duration)/2


        for b in range(splitblocks):
            rStart=b*block
            rEnd=(b+1)*block

            if needleE>rStart and needleE<rEnd:

                return b

        return splitblocks-1

    def spreadBlocks(self,blocks,last):
        print("UNALTERED BLOCKS",blocks)
        spreaded=[]
        for i,b in enumerate(blocks):
            if i==0 and not b:
                #first block is empty
                spreaded.append(last)
            if i==0 and b:
                spreaded.append(b)

            if i>0 and not b:
                spreaded.append(last)

            if i>0 and b:
                spreaded.append(b)

            last=spreaded[i]

        return spreaded

    def validDayblocks(self,dayblocks):

        for b in dayblocks:
            if b is False:
                return False
            else:
                return True
                if b["type"] is None:
                    return False
        return True

    def stripTags(self,text):
        cleanr = re.compile('<.*?>')
        cleantext = re.sub(cleanr, '', text)
        return cleantext
