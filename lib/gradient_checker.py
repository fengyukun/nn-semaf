#! /usr/bin/env python3
"""
Authors: fengyukun
Date:   2016-05-16
Brief:  The gradient checker
"""

from inc import*


class GradientChecker(object):
    """
    Gradient checker class
    """
    def __init__(self, epsilon=1e-04):
        """
        Init GradientChecker
        epsilon: float
            used during checking
        """

        self.epsilon = epsilon

    def check_jagged_input(self, obj, x):
        """
        Checking gradients on  previous layer of given layer for jagged
        input data
        obj: Layer
            Object to check. obj must provide its forward, backprop.
            GradientChecker will use forward function and construct a
            simple loss function to test obj's gradients.
        x: 3d array-like, jagged array
            The input data
        """

        logging.info("To check the gradients on input for jagged x")

        param = np.copy(x)
        gradient_problem = "No problems"
        for t in range(0, len(param)):
            for i in range(0, len(param[t])):
                for val_idx in range(0, len(param[t][i])):
                    estimated_gradient = (
                        self.estimated_general_layer_gradients(
                            obj, param, param[t][i], val_idx
                        )
                    )

                    # Backprop gradients
                    forward_out = obj.forward(param)
                    # Gradients on output units of obj
                    gparam = obj.backprop(forward_out)
                    gradient = gparam[t][i][val_idx]
                    abs_error = abs(gradient - estimated_gradient)
                    if (abs_error > 1e-06):
                        gradient_problem = "HAVE PROBLEMS(WARN)"
                        logging.debug("absolute error:%s" % abs_error)
                        logging.info(
                            "Finish to check, gradients:%s" % gradient_problem
                        )
                        return

        logging.info("Finish to check, gradients:%s" % gradient_problem)
        return

    def check_layer_input(self, obj, x):
        """
        Checking gradients on previous layer of given layer
        obj: Layer
            Object to check. obj must provide its forward, backprop.
            GradientChecker will use forward function and construct a
            simple loss function to test obj's gradients.
        x: numpy.ndarray
            The input data
        """

        logging.info("To check the gradients on input")

        param = np.copy(x)
        it = np.nditer(param, flags=['multi_index', 'refs_ok'],
                       op_flags=['readwrite'])
        gradient_problem = "No problems"
        while not it.finished:
            val_idx = it.multi_index
            val_bak = param[val_idx]
            estimated_gradient = self.estimated_general_layer_gradients(
                obj, param, param, val_idx
            )

            # Backprop gradients
            forward_out = obj.forward(param)
            # Gradients on output units of obj
            gparam = obj.backprop(forward_out)
            gradient = gparam[val_idx]
            abs_error = abs(gradient - estimated_gradient)
            if (abs_error > 1e-06):
                gradient_problem = "HAVE PROBLEMS(WARN)"
                break
            it.iternext()

        logging.info("Finish to check, gradients:%s" % gradient_problem)
        return

    def real_general_layer_gradients(self, obj, x, param, val_idx,
                                     param_index):
        """
        Computing real gradients for general layer at val_idx of param
        obj: object
        param_index: int
            The index of param in the params of obj
        """
        # Backprop gradients
        forward_out = obj.forward(x)
        # Gradients on output unit onf obj
        go = np.copy(forward_out)
        obj.backprop(go)
        gparam = obj.gparams[param_index]
        gradient = gparam[val_idx]
        return gradient

    def estimated_general_layer_gradients(self, obj, x, param, val_idx):
        """
        Computing estimated gradients on general layer at the index of param
        obj: Layer
        x: numpy.ndarray
            x is the input data of obj
        param: numpy.ndarray
        val_idx:
        """
        val_bak = param[val_idx]
        # Estimated gradients
        param[val_idx] += self.epsilon
        forward_out = obj.forward(x)
        inc_loss = 0
        for sample in forward_out:
            for unit in sample:
                inc_loss += ((np.asarray(unit) ** 2) * 0.5).sum()

        param[val_idx] = val_bak
        param[val_idx] -= self.epsilon
        forward_out = obj.forward(x)
        dec_loss = 0
        for sample in forward_out:
            for unit in sample:
                dec_loss += ((np.asarray(unit) ** 2) * 0.5).sum()
        estimated_gradient = (inc_loss - dec_loss) / (2 * self.epsilon)
        # Recover
        param[val_idx] = val_bak

        return estimated_gradient

    def check_layer_params(self, obj, x, check_params=None):
        """
        Checking gradients on parameters of given layer
        obj: Layer
            Object to check. obj must provide its params, param_names, forward,
            backprop. GradientChecker will use forward function and construct a
            simple loss function to test obj's gradients.
        x: numpy.ndarray
            x is the input data of obj
        check_params: list of string
            Parameter names in check_params will be checked. If it is None, all
            parameters will be checked
        """

        logging.info("To check the gradients on parameters")

        if check_params is None:
            check_params = [param_name for param_name in obj.param_names]
        for param_name in check_params:
            param_index = obj.param_names.index(param_name)
            param = obj.params[param_index]
            it = np.nditer(param, flags=['multi_index'],
                           op_flags=['readwrite'])
            gradient_problem = "No problems"
            while not it.finished:
                val_idx = it.multi_index
                val_bak = param[val_idx]
                estimated_gradient = self.estimated_general_layer_gradients(
                    obj, x, param, val_idx
                )

                # Backprop gradients
                gradient = self.real_general_layer_gradients(
                    obj, x, param, val_idx, param_index
                )
                abs_error = abs(gradient - estimated_gradient)
                if (abs_error > 1e-06):
                    gradient_problem = "HAVE PROBLEMS(WARN)"
                    logging.debug("absolute error:%s" % abs_error)
                    break
                it.iternext()
            logging.info("Checking %s, gradient: %s"
                         % (param_name, gradient_problem))

        logging.info("Finish to check gradients")

        return

    def check_nn(self, nn, x, y, check_params=None):
        """
        Check nn 
        x: numpy.ndarray
            x is the input data of nn
        x: numpy.ndarray
            y is the right label of nn
        check_params: list of string
            Parameter names in check_params will be checked. If it is None, all
            parameters will be checked
        """

        if check_params is None:
            check_params = [param_name for param_name in nn.param_names]
        for param_name in check_params:
            param_index = nn.param_names.index(param_name)
            param = nn.params[param_index]
            it = np.nditer(param, flags=['multi_index'],
                           op_flags=['readwrite'])
            gradient_problem = "No problems"
            while not it.finished:
                val_idx = it.multi_index
                val_bak = param[val_idx]
                # Estimated gradients
                param[val_idx] += self.epsilon
                inc_loss = nn.cost(x, y)
                param[val_idx] = val_bak
                param[val_idx] -= self.epsilon
                dec_loss = nn.cost(x, y)
                estimated_gradient = (inc_loss - dec_loss) / (2 * self.epsilon)

                # Backprop gradients
                param[val_idx] = val_bak
                nn.forward(x)
                nn.backprop(y)
                gradient = nn.gparams[param_index][val_idx]
                abs_error = abs(gradient - estimated_gradient)
                if (abs_error > 1e-06):
                    gradient_problem = "HAVE PROBLEMS(WARN)"
                    print(abs_error)
                    break
                it.iternext()
            logging.info("Checking %s, gradient: %s"
                         % (param_name, gradient_problem))

        logging.info("Finish to check gradients")

        return
