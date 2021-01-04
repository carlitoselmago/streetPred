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

    def centerOfLocs(self,points):
        lat = []
        long = []
        for l in points :
          lat.append(l[0])
          long.append(l[1])

        center=((sum(lat)/len(lat)),(sum(long)/len(long)))
        return center


    def buildPredictedHtml(self,center,texts,points,routes=False):
        HTML="""
        <!DOCTYPE html><html lang="es">

        <head>
          <meta charset="UTF-8">
          <title>Predicción mario santamaria</title>
          <link href="https://fonts.googleapis.com/css?family=Open+Sans:400,700" rel="stylesheet">

          <style>
          body{
             font-family: "Open Sans";
          }
            #map_canvas {
              height: 60vw;
              width: 100%;
              margin: 0px;
              padding: 0px;

            }
            #text{
                padding:0px; 15px;
            }
            .numero{
                    background-color: #EA4335;
                    padding: 9px;
                    border-radius: 99999999999999999px;
                    height: 20px;
                    width: 22px;
                    display: inline-block;
                    text-align: center;
                    line-height: 22px;
                    margin-right: 12px;
            }
          </style>

          <script src="https://maps.googleapis.com/maps/api/js?key=AIzaSyDfnB2Y-7gknrP8F2npzaX6GjhOkhHqwJE"></script>
          <script>

          function drawRoute(map,legs){
            var polyline = new google.maps.Polyline({
              path: [],
              strokeColor: "#0000FF",
              strokeWeight: 6
              });
              var bounds = new google.maps.LatLngBounds();

              console.log(legs);
              for (i = 0; i < legs.length; i++) {
              var steps = legs[i].steps;
              for (j = 0; j < steps.length; j++) {
                //TODO: fix this
                var newcord=new google.maps.LatLng(steps[j].end_location.lat, steps[j].end_location.lng)
                polyline.getPath().push(newcord);
                //bounds.extend(steps[j]);
              if (steps[j].hasOwnProperty("path")){
                var nextSegment = steps[j].path;
                for (k = 0; k < nextSegment.length; k++) {
                  polyline.getPath().push(nextSegment[k]);
                  bounds.extend(nextSegment[k]);
                }
                }
              }
              }

              polyline.setMap(map);
          }

            function initialize() {
              var map = new google.maps.Map(
                document.getElementById("map_canvas"), {
        """
        HTML+='center: new google.maps.LatLng('+str(center[0])+', '+str(center[1])+'),'+"\n"
        HTML+='zoom: 13,'+"\n"
        HTML+="""
                  mapTypeId: google.maps.MapTypeId.ROADMAP
                });

                //DRAW ROUTE


        """
        for r in routes:
            HTML+='drawRoute(map,'+r+');'+"\n"


        for i,p in enumerate(points):
             HTML+='new google.maps.Marker({position: {lat:'+str(p[0])+',lng: '+str(p[1])+'},label: String('+str(i+1)+'),map: map});'+"\n"

        HTML+="""
          };

            google.maps.event.addDomListener(window, "load", initialize);
          </script>
        </head>

        <body onload="initialize()">

          <div id="map_canvas"></div>
          <div id="text">
        """

        for l in texts:
            HTML+=l+"<br>"


        HTML+='</div></body></html>'



        Html_file= open("predicted/borrame.html","w")
        Html_file.write(HTML)
        Html_file.close()
