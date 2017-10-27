import json
import pprint

file = open("./input/bellmanFord.json")
output = open("./input/bellmanFord.txt", "w")
getstr = file.read()
getjson = json.loads(getstr)
nedges = 0
pprint.pprint(getjson)
output.writelines(str(getjson["n"]) + "\n")

for keys in getjson["graph"]:
    nedges = nedges + len(getjson["graph"][keys])

output.writelines(str(nedges) + "\n")

for i in getjson["graph"]:
    output.write(i + " ")
    output.write(str(len(getjson["graph"][i])) + " ")
    for j in getjson["graph"][i]:
        nedges = nedges + 1
        output.write("{} {} ".format(j[0], j[1]))
    output.write("\n")