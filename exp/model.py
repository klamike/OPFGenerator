import torch
from torch import Tensor

from ml4opf.layers import BoundRepair

from ml4opf.models.basic_nn.acopf_basic_nn import ACBasicNN, ACBasicNeuralNet


class AC_PGPFVA_BasicNN(ACBasicNN):
    def add_boundrepair(self, boundrepair: str):
        if boundrepair == "none" or boundrepair is None: return

        lower = torch.full((self.output_size,), -torch.inf)
        upper = torch.full((self.output_size,), torch.inf)

        lower[self.pg_slice] = self.violation.pgmin
        upper[self.pg_slice] = self.violation.pgmax
        lower[self.pf_slice] = -self.violation.smax
        upper[self.pf_slice] = self.violation.smax

        self.layers.append(BoundRepair(lower, upper, boundrepair))

    @property
    def pf_slice(self):
        return self.opfmodel.slices[1]["primal/pf"]

class AC_PGPFVA_NeuralNet(ACBasicNeuralNet):
    model: AC_PGPFVA_BasicNN

    def predict(self, pd: Tensor, _: Tensor=None) -> dict[str, Tensor]:
        self.model.eval()

        if len(pd.shape) == 1:
            pd = pd.unsqueeze(0)

        y_hat = self.model.forward(pd)

        pg = y_hat[:, self.pg_slice]
        pf = y_hat[:, self.pf_slice]
        va = y_hat[:, self.va_slice]

        ret: dict[str, Tensor] = dict()
        ret["pg"] = pg
        ret["pf"] = pf
        ret["va"] = va

        return ret
