import torch

class loss:
    def get_disc_loss(self, real_X, fake_x, disc_X, adv_criterion):
       '''
       return the loss of the discriminator given inputs.
       :parameter:
       :param real_X: the real images from pile X
       :param fake_x: the generated images of class X
       :param disc_X: the discriminator for class X; takes images and return real/fake class X
            predict matrices
       :param adv_criterion: the adversarial loss function; takes the discriminator predictions and the target labels and return a adversarial loss (which you aim to minimize)
       :return: return a scaler
       '''
       disc_fake_X_hat = disc_X(fake_x.detach())
       disc_fake_X_loss = adv_criterion(disc_fake_X_hat, torch.zeros_like(disc_fake_X_hat))
       disc_real_X_hat = disc_X(real_X)
       disc_real_X_loss = adv_criterion(disc_real_X_hat, torch.zeros_like(disc_real_X_hat))
       disc_loss = (disc_fake_X_loss + disc_real_X_loss) / 2
       return disc_loss

    def get_gen_adversarial_loss(self, real_X, disc_Y, gen_XY, adv_criterion):
        '''
        return the adversarial loss of the generator given inputs (and the generated images for testing purposes).
        :param real_X: the real image pile X
        :param disc_Y: the discriminator for class Y; takes images and returns real/fake class Y prediction matrices
        :param gen_XY: the generator for class X to Y; takes images and returns the images transformed to class Y
        :param adv_criterion: the adversarial loss function; takes the discriminator predictions and the target labels and returns a adversarial loss (which you aim to minimize)
        :return: return scaler
        '''
        fake_Y = gen_XY(real_X)
        disc_fake_Y_hat = disc_Y(fake_Y)
        adversarial_loss = adv_criterion(disc_fake_Y_hat, torch.zeros_like(disc_fake_Y_hat))
        return adversarial_loss, fake_Y

    def get_indentity_loss(self, real_X, gen_YX, identity_criterion):
        '''
        teturn th identity loss of the generator given inputs (and the generated images for testing purposes).
        :param real_X: the real image from pile X
        :param gen_YX: the generator for class Y to X; takes images and return the images
        transformed to class X
        :param identity_criterion: the identity loss function; takes the real images from X and those images put through a Y->X generator and return th identity loss
        :return:
        '''
        identity_X = gen_YX(real_X)
        identity_loss = identity_criterion(identity_X, real_X)
        return identity_loss, identity_X

    def get_cycle_consistency_loss(self, real_X, fake_Y, gen_YX, cycle_criterion):
        '''
        return the cycle consistency loss of the generator given inputs
        (and the generated images for testing purposes).
        :param real_X: the real images from pile X
        :param fake_Y: the generated images of class Y
        :param gen_YX: the generator for class Y to X; takes images and return the images
        transformed to class X
        :param cycle_criterion: the cycle consistency loss function; takes the real images from X and
        those images put through a X->Y generator and then y->X generator and returns the cycle consistency loss
        :return:
        '''
        cycle_X = gen_YX(fake_Y)
        cycle_loss = cycle_criterion(cycle_X, real_X)
        return cycle_loss, cycle_X

    def gen_gen_loss(self, real_A, real_B, gen_AB, gen_BA, disc_A, disc_B, adv_criterion, identity_criterion, cycle_criterion, lambda_identity=0.1, lambda_cycle=10):
        '''
        Return loss of the generator given inputs.
        :param real_A: the real image from pile A
        :param real_B: the real image from pile B
        :param gen_AB: the generator for class A to B; takes images and return the image transformed to class A
        :param disc_A: the discriminator for class A; takes images and return real/fake class A prediction matrices
        :param disc_B: the discriminator for class B; takes image and return real/fake class B prediction matrices
        :param adv_criterion: the adversarial loss function; takes the discriminator
            predictions and the true labels and returns a adversarial
        :param identity_criterion: he reconstruction loss function used for identity loss
            and cycle consistency loss; takes two sets of images and returns
            their pixel differences (which you aim to minimize)
        :param cycle_criterion: the cycle consistency loss function; takes the real images from X and
            those images put through a X->Y generator and then Y->X generator
            and returns the cycle consistency loss (which you aim to minimize).
            Note that in practice, cycle_criterion == identity_criterion == L1 loss
        :param lambda_identity: the weight of the identity loss
        :param lambda_cycle: the weight of the cycle-consistency loss
        :return:
        '''
        adv_loss_BA, fake_A = self.get_gen_adversarial_loss(real_B, disc_A, gen_BA, adv_criterion)
        adv_loss_AB, fake_B = self.get_gen_adversarial_loss(real_A, disc_B, gen_AB, adv_criterion)
        gen_adversarial_loss = adv_loss_BA + adv_loss_AB

        identity_loss_A, identity_A = self.get_indentity_loss(real_A, gen_BA, identity_criterion)
        identity_loss_B, identity_B = self.get_indentity_loss(real_B, gen_AB, identity_criterion)
        gen_identity_loss = identity_loss_A + identity_loss_B

        cycle_loss_BA, cycle_A = self.get_cycle_consistency_loss(real_A, fake_B, gen_BA, cycle_criterion)
        cycle_loss_AB, cycle_B = self.get_cycle_consistency_loss(real_B, fake_A, gen_AB, cycle_criterion)
        gen_cycle_loss = cycle_loss_BA + cycle_loss_AB

        gen_loss = lambda_identity * gen_identity_loss + lambda_cycle * gen_cycle_loss + gen_adversarial_loss
        return gen_loss, fake_A, fake_B