

def get_values(filename):
  with open(filename, 'r') as document:
    answer = {}
    for line in document:
        line = line.split()
        if not line:  # empty line?
            continue
        try:
            answer[line[0]] = float(line[1:][0])
        except:
            answer[line[0]] = (line[1:][0])
  return answer
