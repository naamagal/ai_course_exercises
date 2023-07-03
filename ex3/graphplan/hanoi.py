import sys

def create_prop(d_idx, p_idx, is_in = True):
    prop = "d%s" % d_idx
    if is_in:
        prop += "in"
    else:
        prop += "out"
    prop += "p%s" % p_idx
    return prop

def create_all_prop(n_disks, m_pegs):
    all_prop_str = ""
    for d_idx in range(0, n_disks):
        for p_idx in range(0, m_pegs):
            prop_str = create_prop(d_idx, p_idx, True)
            all_prop_str += prop_str + " "
            prop_str = create_prop(d_idx, p_idx, False)
            all_prop_str += prop_str + " "
    return all_prop_str


class CreateMoveAct:
    def __init__(self, d_idx, p_from, p_to, n_, m_):
        self.d_idx = d_idx
        self.p_from = p_from
        self.p_to = p_to
        self.n = n_
        self.m = m_

        self.name = "MV_d%sp%sp%s" % (d_idx, p_from, p_to)
        self.pre_list = self.init_pre()
        self.add_list = self.init_add()
        self.del_list = self.init_del()

    #    self.initialized = True

    # def set_nDisk_mPegs(self, nDisks, mPegs):
    #     if not self.initialized:
    #         __nDisks = nDisks
    #         __mPegs = mPegs

    def init_pre(self):
        pre_list = []
        # d_idx in P_from
        pre_list.append(create_prop(self.d_idx, self.p_from, True))
        # d_idx out other pegs
        for p_idx in range(0, self.m):
            if p_idx != self.p_from:
                pre_list.append(create_prop(self.d_idx, p_idx, False))
        # for all d_idx2 < d_idx: d_idx2 out P_to AND d_idx2 out P_from
        for d_idx_sml in range(0, self.d_idx):
            pre_list.append(create_prop(d_idx_sml, self.p_to, False))
            pre_list.append(create_prop(d_idx_sml, self.p_from, False))
        return pre_list

    def init_add(self):
        add_list = []
        # add d_idx in p_to
        add_list.append(create_prop(self.d_idx, self.p_to, True))
        # add d_idx out p_from
        add_list.append(create_prop(self.d_idx, self.p_from, False))
        return add_list

    def init_del(self):
        del_list = []
        # delete d_idx out p_to
        del_list.append(create_prop(self.d_idx, self.p_to, False))
        # delete d_idx in p_from
        del_list.append(create_prop(self.d_idx, self.p_from, True))
        return del_list

    def get_str_name(self):
        return self.name

    def get_str_pre_list(self):
        return " ".join(self.pre_list)

    def get_str_add_list(self):
        return " ".join(self.add_list)

    def get_str_del_list(self):
        return " ".join(self.del_list)


def create_domain_file(domain_file_name, n_, m_):
    # disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    # pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]

    domain_file = open(domain_file_name, 'w')  # use domain_file.write(str) to write to domain_file
    "*** YOUR CODE HERE ***"
    dom_str = "Propositions:\n"
    dom_str += create_all_prop(n_, m_) + "\n"

    dom_str += "Actions:\n"
    # CreateMoveAct.set_nDisk_mPegs(n_, m_)
    for d_idx in range(0, n_):
        for p_from in range(0, m_):
            for p_to in range(0, m_):
                if p_from != p_to:
                    act = CreateMoveAct(d_idx, p_from, p_to, n_, m_)
                    dom_str += "Name: " + act.get_str_name() + "\n"
                    dom_str += "pre: " + act.get_str_pre_list() + "\n"
                    dom_str += "add: " + act.get_str_add_list() + "\n"
                    dom_str += "delete: " + act.get_str_del_list() + "\n"

    domain_file.write(dom_str)
    domain_file.close()


def create_problem_file(problem_file_name_, n_, m_):
    #disks = ['d_%s' % i for i in list(range(n_))]  # [d_0,..., d_(n_ - 1)]
    #pegs = ['p_%s' % i for i in list(range(m_))]  # [p_0,..., p_(m_ - 1)]

    problem_file = open(problem_file_name, 'w')  # use problem_file.write(str) to write to problem_file

    "*** YOUR CODE HERE ***"
    prob_str = "Initial state:"
    for p_idx in range(0, m_):
        for d_idx in range(0, n_):
            # all disk on p0
            if p_idx == 0:
                prob_str += " " + create_prop(d_idx, p_idx, True)
            # all disks out of p1, p2,..., pm
            else:
                prob_str += " " + create_prop(d_idx, p_idx, False)

    prob_str += "\nGoal state:"
    p_last = m_-1
    for p_idx in range(0, m_):
        for d_idx in range(0, n_):
            # all disk on p_m-1 (last peg)
            if p_idx == p_last:
                prob_str += " " + create_prop(d_idx, p_idx, True)
            # all disks out of p1, p2,..., pm
            else:
                prob_str += " " + create_prop(d_idx, p_idx, False)

    problem_file.write(prob_str)
    problem_file.close()


if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: hanoi.py n m')
        sys.exit(2)

    n = int(float(sys.argv[1]))  # number of disks
    m = int(float(sys.argv[2]))  # number of pegs

    domain_file_name = 'hanoi_%s_%s_domain.txt' % (n, m)
    problem_file_name = 'hanoi_%s_%s_problem.txt' % (n, m)

    create_domain_file(domain_file_name, n, m)
    create_problem_file(problem_file_name, n, m)
