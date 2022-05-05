from flask import Flask, request
from proj import *

app = Flask(__name__)
tree = None



data = pd.read_csv("mp_routes.csv")
new_table = MyPyTable(data)
difficulties = data['Rating'].tolist()
stars = data['Avg Stars'].tolist()
type = data['Route Type'].tolist()
pitches = data['Pitches'].tolist()  #clean for greater than x

#alpine
#ice
#Rating
#Length

#star to int
for index,star in enumerate(stars):
    stars[index] = round_trad(star)
for index,pitch in enumerate(pitches):
    pitches[index] = round_30(pitch)

print(stars[0])
print(pitches[0])
print(get_uniques(pitches))


isTrad = []
isSport = []
isAlpine = []
isIce = []
isAid = []
isTrad, isSport = accTradSport(type)
isAlpine, isIce, isAid = accAlpineIceAid(type)
physicalities, Danger = parseDiff(difficulties)
header = ['trad', 'sport', 'pitches', 'alpine', 'ice', 'aid', 'physicalities', 'danger']
X = list(zip(isTrad, isSport, pitches, isAlpine, isIce, isAid, physicalities,Danger))


print('what1')
forest1 = forest(1, 1, 12)
print('what2')
forest1.fit(X, stars, 0.9)
print('what3')

working = "awd"
typeTrad = 'notTrad'
typeSport = 'isSport'
typePitch = 1
typeAlpine = 'notAlpine'
typeIce = 'notIce'
typeAid = 'notAid'
typeDif = '5.10'
typeDanger = 'none'
forest1.predict([[typeTrad, typeSport, 1, typeAlpine, typeIce, typeAid,typeDif,typeDanger]])
forest1.predict([[typeTrad, typeSport, 2, typeAlpine, typeIce, typeAid,typeDif,typeDanger]])

@app.route("/", methods=["GET"])
def index():
    return "<h1>working</h1>", 200

@app.route("/boop", methods=["GET"])
def awd():
    pitch = int(request.args.get('pitch'))
    typeAlpine = str(request.args.get('alpine'))
    typeIce = str(request.args.get('ice'))
    typeAid = str(request.args.get('aid'))
    typeDanger = str(request.args.get('danger'))
    return {'prediction':
                ["trad"] +
                forest1.predict([["isTrad", "notSport", pitch, typeAlpine, typeIce, typeAid,"5.6", typeDanger]]) +
                forest1.predict([["isTrad", "notSport", pitch, typeAlpine, typeIce, typeAid,"5.7",typeDanger]]) +
                forest1.predict([["isTrad", "notSport", pitch, typeAlpine, typeIce, typeAid,"5.8",typeDanger]]) +
                forest1.predict([["isTrad", "notSport", pitch, typeAlpine, typeIce, typeAid,"5.9",typeDanger]]) +
                forest1.predict([["isTrad", "notSport", pitch, typeAlpine, typeIce, typeAid,"5.10",typeDanger]]) +
                forest1.predict([["isTrad", "notSport", pitch, typeAlpine, typeIce, typeAid,"5.11",typeDanger,]]) +
                forest1.predict([["isTrad", "notSport", pitch, typeAlpine, typeIce, typeAid,"5.12",typeDanger]]) +
                forest1.predict([["isTrad", "notSport", pitch, typeAlpine, typeIce, typeAid,"5.other",typeDanger]]) +
                ["sport"] +
               forest1.predict([["notTrad", "isSport", pitch, typeAlpine, typeIce, typeAid,"5.6",typeDanger]]) +
               forest1.predict([["notTrad", "isSport", pitch, typeAlpine, typeIce, typeAid,"5.7",typeDanger]]) +
               forest1.predict([["notTrad", "isSport", pitch, typeAlpine, typeIce, typeAid,"5.8",typeDanger]]) +
               forest1.predict([["notTrad", "isSport", pitch, typeAlpine, typeIce, typeAid,"5.9",typeDanger]]) +
               forest1.predict([["notTrad", "isSport", pitch, typeAlpine, typeIce, typeAid,"5.10",typeDanger]]) +
               forest1.predict([["notTrad", "isSport", pitch, typeAlpine, typeIce, typeAid,"5.11",typeDanger]]) +
               forest1.predict([["notTrad", "isSport", pitch, typeAlpine, typeIce, typeAid,"5.12",typeDanger]]) +
               forest1.predict([["notTrad", "isSport", pitch, typeAlpine, typeIce, typeAid,"5.other",typeDanger]])}, 200

@app.route("/predict", methods=["GET"])
def predict():
    return {
               "prediction": [
                   "trad",
                   "5.6: 2",
                   "5.7: 2",
                   "5.8: 2",
                   "5.9: 2",
                   "5.10: 3",
                   "5.11: 3",
                   "5.12: 3",
                   "5.other: 3",
                   "sport",
                   "5.6: 2",
                   "5.7: 2",
                   "5.8: 2",
                   "5.9: 2",
                   "5.10: 3",
                   "5.11: 2",
                   "5.12: 3",
                   "5.other: 2",
               ]
           }, 200

if __name__ == "__main__":
    app.run(debug=True)