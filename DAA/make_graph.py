import json
import pprint

file = open("./bellmanFord.json")
output = open("./bellmanFord.txt", "w")
getstr = file.read()
getjson = json.loads(getstr)
pprint.pprint(getjson)
output.writelines(str(getjson["n"]) + "\n")
for i in getjson["graph"]:
    output.write(i + " ")
    output.write(str(len(getjson["graph"][i])) + " ")
    for j in getjson["graph"][i]:
        output.write("{} {} ".format(j[0], j[1]))
    output.write("\n")