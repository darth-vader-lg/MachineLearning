
namespace TestChoice
{
   partial class PageIrisKMeans
   {
      /// <summary> 
      /// Variabile di progettazione necessaria.
      /// </summary>
      private System.ComponentModel.IContainer components = null;

      /// <summary> 
      /// Pulire le risorse in uso.
      /// </summary>
      /// <param name="disposing">ha valore true se le risorse gestite devono essere eliminate, false in caso contrario.</param>
      protected override void Dispose(bool disposing)
      {
         if (disposing && (components != null)) {
            components.Dispose();
         }
         base.Dispose(disposing);
      }

      #region Codice generato da Progettazione componenti

      /// <summary> 
      /// Metodo necessario per il supporto della finestra di progettazione. Non modificare 
      /// il contenuto del metodo con l'editor di codice.
      /// </summary>
      private void InitializeComponent()
      {
         this.buttonTrain = new System.Windows.Forms.Button();
         this.labelPrediction = new System.Windows.Forms.Label();
         this.SuspendLayout();
         // 
         // buttonTrain
         // 
         this.buttonTrain.Location = new System.Drawing.Point(3, 3);
         this.buttonTrain.Name = "buttonTrain";
         this.buttonTrain.Size = new System.Drawing.Size(75, 23);
         this.buttonTrain.TabIndex = 1;
         this.buttonTrain.Text = "Train";
         this.buttonTrain.UseVisualStyleBackColor = true;
         this.buttonTrain.Click += new System.EventHandler(this.buttonTrain_Click);
         // 
         // labelPrediction
         // 
         this.labelPrediction.AutoSize = true;
         this.labelPrediction.Location = new System.Drawing.Point(4, 33);
         this.labelPrediction.Name = "labelPrediction";
         this.labelPrediction.Size = new System.Drawing.Size(16, 13);
         this.labelPrediction.TabIndex = 2;
         this.labelPrediction.Text = "...";
         // 
         // PageIrisKMeans
         // 
         this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
         this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
         this.Controls.Add(this.labelPrediction);
         this.Controls.Add(this.buttonTrain);
         this.Name = "PageIrisKMeans";
         this.Size = new System.Drawing.Size(306, 113);
         this.ResumeLayout(false);
         this.PerformLayout();

      }

      #endregion

      private System.Windows.Forms.Button buttonTrain;
      private System.Windows.Forms.Label labelPrediction;
   }
}
