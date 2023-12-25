###############################################################################
# 曲面近似プログラム
###############################################################################
# ==============================================================================
# ライブラリインポート
# ==============================================================================
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# ==============================================================================
# 曲面作成
# ==============================================================================
x, y, z = 10 * np.random.random((3, 10))  # 乱数で適当に x, y, z を作成する
xi, yi = np.linspace(x.min(), x.max(), 100), np.linspace(y.min(), y.max(), 100)  # x, y の値の最小・最大から等間隔の配列を作成する
xi, yi = np.meshgrid(xi, yi)  # xi, yi から mesh を作成する

# カーネル種類
#   線形 RBF => linear
#   ガウシアン RBF => gaussian
#   多重二乗 RBF => multiquadric
#   逆二乗 RBF => inverse
#   多重調和スプライン RBF => cubic(3次), quintic(5次), thin_plate(薄板スプライン)
rbf = interpolate.Rbf(x, y, z, function='gaussian')  # x, y, z の値で RBF 補間をして曲面を作成する

zi = rbf(xi, yi)  # x, y, z の曲面における xi, yi の位置の zi を計算する

# ==============================================================================
# 3 次元グラフ
# ==============================================================================
fig = plt.figure(figsize=(18, 7), dpi=200)  # 画像を作成する
el = 55  # 視点高さを設定する

# ==============================================================================
ax = fig.add_subplot(231, projection='3d')  # 2 × 3 の 1 枚目に描画する

# 曲面のプロット
ax.plot_surface(xi, yi, zi)  # サーフェスの描画
ax.view_init(elev=el, azim=10)  # ビューの設定
ax.set_title('elev = ' + str(el) + ', deg = 10')  # タイトルの設定
ax.set_xlabel('xi')  # 軸ラベルの設定
ax.set_ylabel('yi')  # 軸ラベルの設定
ax.set_zlabel('zi')  # 軸ラベルの設定

# ==============================================================================
ax = fig.add_subplot(232, projection='3d')  # 2 × 3 の 2 枚目に描画する

# 曲面のプロット
ax.plot_surface(xi, yi, zi)  # サーフェスの描画
ax.view_init(elev=el, azim=45)  # ビューの設定
ax.set_title('elev = ' + str(el) + ', deg = 45')  # タイトルの設定
ax.set_xlabel('xi')  # 軸ラベルの設定
ax.set_ylabel('yi')  # 軸ラベルの設定
ax.set_zlabel('zi')  # 軸ラベルの設定

# ==============================================================================
ax = fig.add_subplot(233, projection='3d')  # 2 × 3 の 3 枚目に描画する

# 曲面のプロット
ax.plot_surface(xi, yi, zi)  # サーフェスの描画
ax.view_init(elev=el, azim=80)  # ビューの設定
ax.set_title('elev = ' + str(el) + ', deg = 80')  # タイトルの設定
ax.set_xlabel('xi')  # 軸ラベルの設定
ax.set_ylabel('yi')  # 軸ラベルの設定
ax.set_zlabel('zi')  # 軸ラベルの設定

# ==============================================================================
ax = fig.add_subplot(234, projection='3d')  # 2 × 3 の 4 枚目に描画する

# 曲面のプロット
#   rstride と cstride はステップサイズ
#   cmap は彩色
#   linewidth は曲面のメッシュの線の太さ
ax.plot_wireframe(xi, yi, zi, rstride=1, cstride=1, cmap='hsv', linewidth=0.2)  # ワイヤーフレームの描画
ax.view_init(elev=el, azim=10)  # ビューの設定
ax.set_xlabel('xi')  # 軸ラベルの設定
ax.set_ylabel('yi')  # 軸ラベルの設定
ax.set_zlabel('zi')  # 軸ラベルの設定

# ==============================================================================
ax = fig.add_subplot(235, projection='3d')  # 2 × 3 の 5 枚目に描画する

# 曲面のプロット
#   rstride と cstride はステップサイズ
#   cmap は彩色
#   linewidth は曲面のメッシュの線の太さ
ax.plot_wireframe(xi, yi, zi, rstride=1, cstride=1, cmap='hsv', linewidth=0.2)  # ワイヤーフレームの描画
ax.view_init(elev=el, azim=45)  # ビューの設定
ax.set_xlabel('xi')  # 軸ラベルの設定
ax.set_ylabel('yi')  # 軸ラベルの設定
ax.set_zlabel('zi')  # 軸ラベルの設定

# ==============================================================================
ax = fig.add_subplot(236, projection='3d')  # 2 × 3 の 6 枚目に描画する

# 曲面のプロット
#   rstride と cstride はステップサイズ
#   cmap は彩色
#   linewidth は曲面のメッシュの線の太さ
ax.plot_wireframe(xi, yi, zi, rstride=1, cstride=1, cmap='hsv', linewidth=0.2)  # ワイヤーフレームの描画
ax.view_init(elev=el, azim=80)  # ビューの設定
ax.set_xlabel('xi')  # 軸ラベルの設定
ax.set_ylabel('yi')  # 軸ラベルの設定
ax.set_zlabel('zi')  # 軸ラベルの設定

# ==============================================================================
# グラフ出力
file_name = 'Various 3D Images.jpg'  # グラフ名設定
plt.savefig(file_name)  # グラフ出力

plt.show()  # 描画

# ==============================================================================
# 2 次元グラフ
# ==============================================================================
fig = plt.figure(figsize=(18, 7), dpi=200)  # 画像を作成する

# 曲面のプロット ==================================================================
ax = fig.add_subplot(121, projection='3d')  # 1 × 2 の 1 枚目に描画する
ax.plot_surface(xi, yi, zi)  # サーフェスの描画
el = 100  # 視点高さを設定する
ax.view_init(elev=el, azim=90)  # ビューの設定
ax.set_title('elev = 100, deg = 90')  # タイトルの設定
ax.set_xlabel('xi')  # 軸ラベルの設定
ax.set_ylabel('yi')  # 軸ラベルの設定
ax.set_zlabel('zi')  # 軸ラベルの設定

# コンター =======================================================================
ax = fig.add_subplot(122)  # 1 × 2 の 2 枚目に描画する
contour = ax.contourf(xi, yi, zi)
fig.colorbar(contour)
ax.set_xlim([x.max(), x.min()])
ax.set_ylim([y.max(), y.min()])
ax.set_title('contour')  # タイトルの設定
ax.set_xlabel('xi')  # 軸ラベルの設定
ax.set_ylabel('yi')  # 軸ラベルの設定

# グラフ出力
file_name = '3D and Contour.jpg'  # グラフ名設定
plt.savefig(file_name)  # グラフ出力

plt.show()