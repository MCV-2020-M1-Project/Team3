Steps to setup environment

- Create a Virtual Environment
virtualenv -p python3 env

- Source the environment
source ./env/bin/activate

- Install dependencies using pip
pip install -r requirements.txt

<h4>FOR WEEK 1</h4>
Run the command 'python main.py --week1' 

Folders BBDD, qsd1_w1, qsd2_w1, qst1_w1 and qst2_w1 MUST be inside a folder named "data". Structure:

- week1
- data
    - BBDD
    - qsd1_w1
    - qsd2_w1
    - qst1_w1
    - qst2_w1
    
<h4>FOR WEEK 2</h4>
Run the command 'python main.py --week2' 

Folders BBDD, qsd2_w1, qsd1_w2, qsd2_w2, qst1_w2 and qst2_w2 MUST be inside a folder named "data". 
IMPORTANT: week1 folder needs to be downloaded as well, as it works as a library with the functions of first week improved.
Folder structure:
- week1
- week2
- data
    - BBDD
    - qsd2_w1
    - qsd1_w2
    - qsd2_w2
    - qst1_w2
    - qst2_w2
