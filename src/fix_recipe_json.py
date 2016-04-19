import json
import sys

## reformatting initial recipe file

def fix_json_formatting(filename, outfilename):
    with open(filename) as infile:
        for line in infile:
            ddict = json.loads(line)
            new_dict = {}
            new_dict['R_id'] = ddict.keys()[0]
            #print ddict.keys()[0]
            #print ddict.values()[0]['publisher']
            new_dict['publisher'] = ddict.values()[0]['publisher']
            new_dict['ingredients'] = ddict.values()[0]['ingredients']
            new_dict['image_url'] = ddict.values()[0]['image_url']
            new_dict['source_url'] = ddict.values()[0]['source_url']
            new_dict['title'] = ddict.values()[0]['title']
            with open(outfilename, 'a') as outfile:
                outfile.write("{}\n".format(json.dumps(new_dict)))

if __name__ == '__main__':
    infile = sys.argv[1]
    outfile = sys.argv[2]

    fix_json_formatting(infile, outfile)
