import json
import os
from pathlib import Path

path = Path(__file__).parent.absolute().parent.absolute()

def determineIdentity():
    maskedImagesLocation = os.path.sep.join([str(path), "MaskedImages"])
    outputLocation = os.path.sep.join([str(path), "CaffeMaskDetection", "identity.json"])

    maskedFilesArray = os.listdir(maskedImagesLocation)

    identities = {}

    for maskedName in maskedFilesArray:
        if (maskedName == "frame.jpg"):
            continue

        fileName = maskedName.split(".")[0]

        identity = str(input("Identity of Person within: " + maskedName + "?\n")).strip()

        identities[fileName] = identity

    with open(outputLocation, "w") as outfile:
        json.dump(identities, outfile)


def getIdentity(fileName):

    identityLocation = os.path.sep.join([str(path), "CaffeMaskDetection", "identity.json"])

    if (not os.path.exists(identityLocation)):
        determineIdentity()
        getIdentity(fileName)

        #return fileName

    with open(identityLocation) as inputFile:
        identities = json.load(inputFile)

    try:
        return identities[fileName]
    except KeyError as keyError:
        return fileName

#determineIdentity()
#print(getIdentity("anthonyMasked"))
