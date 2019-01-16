import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def class_text_to_int(row_label):
    if row_label == 'right':
        return 1
    elif row_label == 'left':
        return 2
    else:
        None

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text),
                     class_text_to_int(member[0].text),
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text)
                    
                     )
            xml_list.append(value)
    column_name = ['image_name', 'xmin', 'ymin', 'xmax', 'ymax', 'class_id', 'width', 'height']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'datasets/Validation')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('Validation.csv', index=None)
    print('Successfully converted xml to csv.')


main()
