from keras.datasets import mnist
from keras.models import load_model
from keras.utils import to_categorical


(X_train, Y_train), (X_test, Y_test) = mnist.load_data()
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255
Y_test = to_categorical(Y_test, 10)


model_mlp = load_model('modele_TP2.h5')
model_cnn = load_model('modele_TP2_CNN.h5')
model_lenet = load_model('modele_TP2_net.h5')

loss_mlp, acc_mlp = model_mlp.evaluate(X_test, Y_test, verbose=0)
loss_cnn, acc_cnn = model_cnn.evaluate(X_test, Y_test, verbose=0)
loss_lenet, acc_lenet = model_lenet.evaluate(X_test, Y_test, verbose=0)


print("\n" + "="*55)
print("📊 COMPARAISON DES 3 MODÈLES SUR MNIST")
print("="*55)
print(f"MLP baseline   : Accuracy = {acc_mlp*100:.2f}%   |   Loss = {loss_mlp:.4f}")
print(f"CNN custom     : Accuracy = {acc_cnn*100:.2f}%   |   Loss = {loss_cnn:.4f}")
print(f"Variante LeNet : Accuracy = {acc_lenet*100:.2f}%   |   Loss = {loss_lenet:.4f}")
print("="*55)