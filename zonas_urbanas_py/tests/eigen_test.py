import numpy as np
import unittest

import zonas_urbanas


class TesteZonasUrbanas(unittest.TestCase):
    matriz_teste = np.array(
        [[2, 1, 0, 0, 1, 0],
         [1, 0, 1, 0, 1, 0],
         [0, 1, 0, 1, 0, 0],
         [0, 0, 1, 0, 1, 1],
         [1, 1, 0, 1, 0, 0],
         [0, 0, 0, 1, 0, 0]]
    )
    res_autovet = np.array(
        [[00.00863434, 00.24230289, -0.33682166, -0.26361110, 00.47642275, -0.72890087]
         [00.43185661, -0.62498438, 00.01708410,
             00.48339092, -0.06739702, -0.42941052]
         [-0.46352632, 00.27565802, -0.42123889,
             00.54176963, -0.44304540, -0.20471992]
         [00.55825151, 00.41573818, 00.10372859, -
             0.24986504, -0.62677819, -0.22242740]
         [-0.46756855, -0.04354843, 00.75316077, -
             0.06979638, -0.13897677, -0.43364266]
         [-0.26134951, -0.54768768, -0.36167158, -0.57967577, -0.40003041, -0.06985681]]
    )
    res_autoval = np.array(
        [[-2.1360343,  -0.75907894, -0.28680327,
            0.43104276,  1.56682634,  3.1840474]]
    )

    def teste_eh_simetrica(self):
        self.assertFalse(zonas_urbanas.eh_simetrica(np.array([[1, 2, 3]])))
        self.assertTrue(zonas_urbanas.eh_simetrica(self.matriz_teste))

    def teste_classe_eigen(self):
        eigen = zonas_urbanas.Eigen(self.matriz_teste)
        eigen.calcular()
        self.assertTrue((self.res_autovet == eigen.auto_vetores()).all())
        self.assertTrue((self.res_autoval == eigen.auto_valores()).all())


if __name__ == '__main__':
    unittest.main()
