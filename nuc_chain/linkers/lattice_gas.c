#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <stdbool.h>

#define NUM_MOVES 4
const double TARGET_ACCEPTANCE = 0.1;
const double KBT = 1;

/* qsort int comparison function */
int int_cmp(const void *a, const void *b)
{
        const int *ia = (const int *)a; // casting pointer types
        const int *ib = (const int *)b;
        return *ia  - *ib;
        /* integer comparison: returns negative if b > a
        *  and positive if a > b */
}

int random_in_range(int n)
{
    long int rv = random();
    double frac = (double)rv/(double)RAND_MAX;
    return (int)(frac*n);
}

void insert_bead(int const *lattice, int *new_lattice, int nbeads, int new_pos)
{
    int i = 0;
    for ( ; i < nbeads && lattice[i] < new_pos; i++) {
        new_lattice[i] = lattice[i];
    }
    new_lattice[i] = new_pos;
    for ( ; i < nbeads; i++) {
        new_lattice[i+1] = lattice[i];
    }
}

int add_move(int const *lattice, int *new_lattice, int nbeads,
        const int max_beads, const int n)
{
    if (nbeads == max_beads) {
        for (int i = 0; i < nbeads; i++) {
            new_lattice[i] = lattice[i];
        }
        return nbeads;
    }
    int new_pos = random_in_range(n);
    insert_bead(lattice, new_lattice, nbeads, new_pos);
    return nbeads + 1;
}

void remove_bead(int const *lattice, int *new_lattice, int nbeads, int del_i)
{
    int i = 0;
    for ( ; i < del_i; i++) {
        new_lattice[i] = lattice[i];
    }
    for (; i < nbeads; i++) {
        new_lattice[i] = lattice[i+1];
    }
}

int remove_move(int const *lattice, int *new_lattice, int nbeads)
{
    if (nbeads == 0) {
        return nbeads;
    }
    int del_i = random_in_range(nbeads);
    remove_bead(lattice, new_lattice, nbeads, del_i);
    return nbeads - 1;

}
int joint_slide_move(int const *lattice, int *new_lattice, int nbeads,
        const int n, double mag)
{
    int left = random_in_range(n);
    int right = random_in_range(n);
    if (right < left) {
        int tmp = left;
        left = right;
        right = tmp;
    }
    int offset = (int) -mag*log(random()/(double)RAND_MAX);
    if (random_in_range(2) == 1) {
        offset *= -1;
    }
    for (int i = 0; i < nbeads; i++) {
        new_lattice[i] = lattice[i];
        if (left <= i && i < right) {
            new_lattice[i] += offset;
            if (new_lattice[i] >= nbeads) {
                new_lattice[i] = nbeads - 1;
            } else if (new_lattice[i] < 0) {
                new_lattice[i] = 0;
            }
        }
    }
    qsort(new_lattice, nbeads, sizeof(int), int_cmp);
    return nbeads;
}

void move_bead(int const *lattice, int *new_lattice, int nbeads, int ind, int new_pos)
{
    // TODO a "jump" based search that uses the offset and assumes packed
    // lattice to make much better initial guess
    int new_ind = ind;
    if (lattice[ind] < new_pos) {
        while (new_ind < nbeads && lattice[new_ind] < new_pos) {
            new_ind++;
        }
        for (int i = 0; i < ind; i++) {
            new_lattice[i] = lattice[i];
        }
        for (int i = ind; i < new_ind - 1; i++) {
            new_lattice[i] = lattice[i+1];
        }
        new_lattice[new_ind-1] = new_pos;
        for (int i = new_ind; i < nbeads; i++) {
            new_lattice[i] = lattice[i];
        }
    } else if (new_pos < lattice[ind]) {
        while (new_ind > 0 && new_pos < lattice[new_ind]) {
            new_ind--;
        }
        for (int i = 0; i < new_ind; i++) {
            new_lattice[i] = lattice[i];
        }
        new_lattice[new_ind] = new_pos;
        for (int i = new_ind + 1; i < ind + 1; i++) {
            new_lattice[i] = lattice[i-1];
        }
        for (int i = ind + 1; i < nbeads; i++) {
            new_lattice[i] = lattice[i];
        }
    }
    /* got too confusing, gonna copy my python code instead
    if (ind == new_ind) {
        for (int i = 0; i < nbeads; i++) {
            new_lattice[i] = lattice[i];
        }
    }
    // now new_ind holds the index we would have to insert before (in [0,
    // new_ind]) to keep array in order
    int i = 0;
    int i_n = 0;
    while (i_n < nbeads) {
        if (i_n == new_ind) {
            new_lattice[i_n++] = new_pos;
            continue;
        }
        if (i == ind) {
            i++;
        }
        new_lattice[i_n++] = lattice[i++];
    }
    */
}

int single_slide_move(int const *lattice, int *new_lattice, int nbeads,
        const int n, double mag)
{
    int choice = random_in_range(nbeads);
    int offset = (int) -mag*log(random()/(double)RAND_MAX);
    if (random_in_range(2) == 1) {
        offset *= -1;
    }
    int new_pos = lattice[choice] + offset;
    if (new_pos < 0) new_pos = 0;
    if (new_pos >= n) new_pos = n-1;
    move_bead(lattice, new_lattice, nbeads, choice, new_pos);
    return nbeads;
}

bool colliding(int const * lattice, int nbeads, int width)
{
    for (int i = 1; i < nbeads; i++) {
        if (lattice[i] - lattice[i-1] < width) {
            return true;
        }
    }
    return false;
}

void print_lattice(int *lattice, int nbeads)
{
    // printf("Lattice<%d>: ", nbeads);
    int i;
    for (i = 0; i < nbeads - 1; i++) {
        printf("%d, ", lattice[i]);
    }
    printf("%d", lattice[i]);
    printf("\n");
}

bool check_lattice_match(int *l1, int *l2, int n)
{
    for (int i = 0; i < n; i++) {
        if (l1[i] != l2[i]) {
            return false;
        }
    }
    return true;
}

void check_test(int *l1, int *l2, int n)
{
    if (!check_lattice_match(l1, l2, n)) {
        print_lattice(l1, n);
        print_lattice(l2, n);
    }
}

void run_tests(void)
{
    int lattice[6] = {2, 5, 17, 21, 33, 0};
    int test_lattice[6];
    int nbeads = 5;

    int expected_lattice_1[6] = {5, 7, 17, 21, 33, 0}; // slide [0]:->7
    move_bead(lattice, test_lattice, nbeads, 0, 7);
    check_test(test_lattice, expected_lattice_1, nbeads);

    int expected_lattice_2[6] = {1, 2, 5, 17, 21, 0}; // slide [4]:->1
    move_bead(lattice, test_lattice, nbeads, 4, 1);
    check_test(test_lattice, expected_lattice_2, nbeads);

    int expected_lattice_3[6] = {2, 17, 21, 33, 42, 0};// slide [1]:->42
    move_bead(lattice, test_lattice, nbeads, 1, 42);
    check_test(test_lattice, expected_lattice_3, nbeads);

    int expected_lattice_4[6] = {1, 2, 5, 17, 21, 33}; // add 1
    insert_bead(lattice, test_lattice, nbeads, 1);
    check_test(test_lattice, expected_lattice_4, nbeads+1);

    int expected_lattice_5[6] = {2, 5, 17, 20, 21, 33};// add 20
    insert_bead(lattice, test_lattice, nbeads, 20);
    check_test(test_lattice, expected_lattice_5, nbeads+1);

    int expected_lattice_6[6] = {2, 5, 17, 21, 33, 35}; // add 35
    insert_bead(lattice, test_lattice, nbeads, 35);
    check_test(test_lattice, expected_lattice_6, nbeads+1);

    int expected_lattice_7[6] = {2, 5, 17, 33, 0, 0}; // remove 3
    remove_bead(lattice, test_lattice, nbeads, 3);
    check_test(test_lattice, expected_lattice_7, nbeads-1);

}

int main(int argc, char *argv[])
{
    /* read inputs */
    if (argc == 2) {
        run_tests();
        return 0;
    }
    if (argc != 6) {
        fprintf(stderr, "usage: %s n_bins particle_width mu num_steps save_interval\n", argv[0]);
        exit(1);
    }
    const int n = strtol(argv[1], NULL, 10);
    const int w = strtol(argv[2], NULL, 10);
    const double mu = strtod(argv[3], NULL);
    const int num_steps = strtol(argv[4], NULL, 10);
    const int save_interval = strtol(argv[5], NULL, 10);
    const int adaptation_interval = (int)fmax(100, fmin(num_steps/100, 10000));
    /* initialize tracking variables */
    int n_accepted[NUM_MOVES] = {0,0,0,0};
    int n_attempted[NUM_MOVES] = {0,0,0,0};
    int j_a = 0; // track adaptation interval
    int j_s = 0; // track save interval
    double params[NUM_MOVES] = {1,1,1,1};
    /* initialize monte carlo state */
    double acc_prob = 0;
    const int max_beads = n/w;
    int lattice[max_beads], new_lattice[max_beads];
    lattice[0] = 0;
    new_lattice[0] = 0;
    for (int k = 1; k < max_beads; k++) {
        lattice[k] = lattice[k-1] + w;
        new_lattice[k] = new_lattice[k-1] + w;
    }
    int nbeads = max_beads;
    int nbeads_new = max_beads;
    srandom((unsigned int)1);
    int rv = 0;
    for (int i = 0; i < num_steps; i++)
    {
        // perform parameter adaptation
        if (j_a == adaptation_interval) {
            printf("Acceptance ratios: ");
            for (int k = 0; k < NUM_MOVES; k++) {
                // make more stringent
                double acceptance_ratio = (double)n_accepted[k]/(double)n_attempted[k];
                printf("%04.8g ", acceptance_ratio);
                if (acceptance_ratio < TARGET_ACCEPTANCE) {
                    params[k]++;
                } else {
                    params[k]--;
                }
                n_accepted[k] = 0;
                n_attempted[k] = 0;
            }
            printf("\n");
            j_a = 0;
        }
        rv = random_in_range(NUM_MOVES);
        switch (rv) {
            case 0:
                nbeads_new = add_move(lattice, new_lattice, nbeads, max_beads, n);
                // new_energy - energy = -1 (one more bead added)
                acc_prob = n/(nbeads+1)*exp(-KBT*(-mu - 1));
                break;
            case 1:
                nbeads_new = remove_move(lattice, new_lattice, nbeads);
                // new_energy - energy = 1 (one bead removed)
                acc_prob = (nbeads/n)*exp(-KBT*(mu + 1));
                break;
            case 2:
                nbeads_new = single_slide_move(lattice, new_lattice, nbeads, n, params[2]);
                acc_prob = 1; // no change in energy
                break;
            case 3:
                nbeads_new = joint_slide_move(lattice, new_lattice, nbeads, n, params[3]);
                acc_prob = 1; // no change in energy
                break;
            default:
                fprintf(stderr, "Invalid move requested.\n");
                exit(1);
        }
        n_attempted[rv]++;
        if (!colliding(new_lattice, nbeads_new, w) && random()/(double)RAND_MAX < acc_prob) {
            n_accepted[rv]++;
            for (int k = 0; k < nbeads_new; k++) {
                lattice[k] = new_lattice[k];
            }
            nbeads = nbeads_new;
        }
        //TODO figure out most satisfyign way to space interval
        // maybe make an option for wehtehr to output initial condition?
        if (j_s == save_interval) {
            print_lattice(lattice, nbeads);
            j_s = 0;
        }
        j_a++; j_s++;
    }

    return 0;
}
