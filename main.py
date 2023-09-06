import os
import numpy as np
import cv2
import numpy as np
from googletrans import Translator
import pandas as pd
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import Model, load_model

app = Flask(__name__)

# Load model_tourism model
model_tourism = load_model('model_tourism.h5')
# Load model_hieroglyphs model
model_hieroglyphs = load_model('model_hieroglyphs.h5')


# Define a function to preprocess an image
# utils_hieroglyphs.p
def get_hieroglyphs_classes():
    class_names = ['100', 'Among', 'Angry', 'Ankh', 'Aroura', 'At', 'Bad_Thinking', 'Bandage', 'Bee', 'Belongs',
                   'Birth', 'Board_Game', 'Book', 'Boy', 'Branch', 'Bread', 'Brewer', 'Builder', 'Bury', 'Canal',
                   'Cloth_on_Pole', 'Cobra', 'Composite_Bow', 'Cooked', 'Corpse', 'Dessert', 'Divide', 'Duck',
                   'Elephant', 'Enclosed_Mound', 'Eye', 'Fabric', 'Face', 'Falcon', 'Fingre', 'Fish', 'Flail',
                   'Folded_Cloth', 'Foot', 'Galena', 'Giraffe', 'He', 'Her', 'Hit', 'Horn', 'King', 'Leg',
                   'Length_Of_a_Human_Arm', 'Life_Spirit', 'Limit', 'Lion', 'Lizard', 'Loaf', 'Loaf_Of_Bread', 'Man',
                   'Mascot', 'Meet', 'Mother', 'Mouth', 'Musical_Instrument', 'Nile_Fish', 'Not', 'Now', 'Nurse',
                   'Nursing', 'Occur', 'One', 'Owl', 'Pair', 'Papyrus_Scroll', 'Pool', 'QuailChick', 'Reed', 'Ring',
                   'Rope', 'Ruler', 'Sail', 'Sandal', 'Semen', 'Small_Ring', 'Snake', 'Soldier', 'Star', 'Stick',
                   'Swallow', 'This', 'To_Be_Dead', 'To_Protect', 'To_Say', 'Turtle', 'Viper', 'Wall', 'Water', 'Woman',
                   'You']

    return class_names


def predict_hieroglyphs(model, processed_image, class_names):
    predicted_class = np.argmax(model.predict(processed_image))
    predicted_class_name = class_names[predicted_class]
    return predicted_class_name


# utils_tourism.py
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))  # Resize the image to match the model input size
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert to RGB
    img = img / 255.0  # Normalize pixel values
    return np.expand_dims(img, axis=0)


def get_tourism_prediction(processed_image, model, data_dict, class_names, lang):
    predicted_class = np.argmax(model.predict(processed_image))

    # Convert class index to class name
    predicted_class_name = class_names[predicted_class]
    final_predict = data_dict[predicted_class_name]
    # Create a Translator object
    translator = Translator()

    # Text to be translated
    text_to_translate = final_predict
    # Translate the text to another language
    if lang != 'en':
        translated_text = translator.translate(text_to_translate, src='en',
                                               dest=lang)
    else:
        translated_text = final_predict

    return translated_text, predicted_class_name


def get_tourism_data():
    # Load the Excel file
    xlsx_path = 'data_set.xlsx'
    data = pd.read_excel(xlsx_path)

    # Assuming the Excel file has columns named "key_column" and "value_column"
    key_column = "exhibits"
    value_column = "Text in English"

    # Convert the data to a dictionary
    data_dict = dict(zip(data[key_column], data[value_column]))
    return data_dict


def get_tourism_classes():
    class_names = ['10_The_HolyQuran', '11_King_Thutmose_III', '12_King_Fouad_I', '13_theVizier_Paser',
                   '14_Sphinxof_theking_Amenemhat_III', '15_Amun_Ra_Kingof_theGods', '16_Nazlet_Khater_Skeleton',
                   '17_Pen_Menkh_TheGovernerOf_Dendara', '18_TheCoffinOf_Lady_Isis', '19_CoffinOf_Nedjemankh',
                   '1_the_female_peasent', '20_TheCoffinOf_Sennedjem', '21_A_silo', '22_Captives_statuettes',
                   '23_Chair_from_the_tomb_of_Queen_Hetepheres', '24_Maat', '25_Mahalawi_water_ewers',
                   '26_Mamluk_Lamps',
                   '27_Khedive_Ismail', '28_Mohamed_Talaat_Pasha_Harb', '29_Model_of_building', '2_statue_ofthe_sphinx',
                   '30_Muhammad_Ali_Pasha', '31_Puplit _of_the_Mosque_of_Abu_Bakr_bin_Mazhar',
                   '32_The_Preist_Psamtik_seneb', '33_The_Madrasaa_and_Mosque_of_Sultan_Hassan', '34_Wekalet_al-Ghouri',
                   '35_The_birth_of_Isis', '36_King_Akhenaten', '37_The_Kiswa_Covering_of_holy_Kaaba',
                   '38_AQueen_in_the_form_of_the_Sphinx', '39_Purification_with_water', '3_Hassan_Fathi',
                   '40_Mashrabiya', '41_Astrolabe', '42_Baker', '43_The_Protective_Godesses', '44_Miller',
                   '45_Hapi_The_Scribe', '46_Thoth', '47_Ottoman_Period_Carpet', '48_Stela_of_King_Qaa',
                   '49_Zainab_Khatun_house', '4_Royal_Statues', '50_God_Nilus', '5_Greek_Statues', '6_Khonsu',
                   '7_Ra_Horakhty', '8_Senenmut', '9_Box_ofthe_Holy Quran', 'Akhenaten', 'Bent pyramid for senefru',
                   'Colossal Statue of Ramesses II', 'Colossoi of Memnon', 'Goddess Isis with her child', 'Hatshepsut',
                   'Hatshepsut face', 'Khafre Pyramid', 'Mask of Tutankhamun', 'Nefertiti', 'Pyramid_of_Djoser',
                   'Ramessum', 'Ramses II Red Granite Statue', 'Statue of King Zoser',
                   'Statue of Tutankhamun with Ankhesenamun', 'Temple_of_Isis_in_Philae', 'Temple_of_Kom_Ombo',
                   'The Great Temple of Ramesses II', 'amenhotep iii and tiye', 'bust of ramesses ii',
                   'menkaure pyramid', 'sphinx']

    return class_names


@app.route('/')
def index():
    return render_template('index.html', prediction=None)


@app.route('/predictTourism', methods=['POST'])
def predict():
    try:
        # Get the image file from the request
        image_file = request.files['image']
        selected_language = request.form['language']
        image_path = "temp_image.jpg"
        image_file.save(image_path)

        # Load tourism data
        data_dict = get_tourism_data()

        # load classes names
        class_names = get_tourism_classes()

        # Load and preprocess the image
        processed_image = preprocess_image(image_path)

        # Make a prediction
        translated_text, predicted_class_name = get_tourism_prediction(processed_image, model_tourism, data_dict,
                                                                       class_names, selected_language)
        if selected_language != 'en':
            translated_text = translated_text.text
        else:
            translated_text = translated_text
        return render_template('index.html', prediction=translated_text)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predictTourismAPI', methods=['POST'])
def predictapi():
    try:
        # Get the image file from the request
        image_file = request.files['image']
        selected_language = request.form['language']
        image_path = "temp_image.jpg"
        image_file.save(image_path)

        # Load tourism data
        data_dict = get_tourism_data()

        # load classes names
        class_names = get_tourism_classes()

        # Load and preprocess the image
        processed_image = preprocess_image(image_path)

        # Make a prediction
        translated_text, predicted_class_name = get_tourism_prediction(processed_image, model_tourism, data_dict,
                                                                       class_names, selected_language)
        if selected_language != 'en':
            translated_text = translated_text.text
        else:
            translated_text = translated_text
        return jsonify({
            "information": translated_text,
            "name": predicted_class_name
        })

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predictHieroglyphs', methods=['POST'])
def predict2():
    try:
        # Get the image file from the request
        image_file = request.files['image']
        image_path = "temp_image.jpg"
        image_file.save(image_path)

        # load classes names
        class_names = get_hieroglyphs_classes()

        # Load and preprocess the image
        processed_image = preprocess_image(image_path)

        # Make a prediction
        prediction = predict_hieroglyphs(model_hieroglyphs, processed_image, class_names)

        return render_template('index.html', prediction=prediction)

    except Exception as e:
        return jsonify({"error": str(e)})


@app.route('/predictHieroglyphsAPI', methods=['POST'])
def predictapi2():
    try:
        # Get the image file from the request
        image_file = request.files['image']
        image_path = "temp_image.jpg"
        image_file.save(image_path)

        # load classes names
        class_names = get_hieroglyphs_classes()

        # Load and preprocess the image
        processed_image = preprocess_image(image_path)

        # Make a prediction
        prediction = predict_hieroglyphs(model_hieroglyphs, processed_image, class_names)

        return jsonify({"class": prediction})

    except Exception as e:
        return jsonify({"error": str(e)})


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
