from roop.typing import Frame

MAX_PROBABILITY = 0.85


def predict_frame(target_frame: Frame) -> bool:
    image = Image.fromarray(target_frame)
    views = numpy.expand_dims(image, axis=0)
    _, probability = model.predict(views)[0]
    return probability > MAX_PROBABILITY
    
def predict_video(target_path: str) -> bool:
    return any(probability > MAX_PROBABILITY for probability in probabilities)
