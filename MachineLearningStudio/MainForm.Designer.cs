
namespace TestChoice
{
   partial class MainForm
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

      #region Codice generato da Progettazione Windows Form

      /// <summary>
      /// Metodo necessario per il supporto della finestra di progettazione. Non modificare
      /// il contenuto del metodo con l'editor di codice.
      /// </summary>
      private void InitializeComponent()
      {
         this.buttonTrainIris = new System.Windows.Forms.Button();
         this.labelPrediction = new System.Windows.Forms.Label();
         this.panelPlot = new System.Windows.Forms.Panel();
         this.buttonPlot = new System.Windows.Forms.Button();
         this.buttonTrainFeet = new System.Windows.Forms.Button();
         this.SuspendLayout();
         // 
         // buttonTrainIris
         // 
         this.buttonTrainIris.Location = new System.Drawing.Point(13, 13);
         this.buttonTrainIris.Name = "buttonTrainIris";
         this.buttonTrainIris.Size = new System.Drawing.Size(75, 23);
         this.buttonTrainIris.TabIndex = 0;
         this.buttonTrainIris.Text = "Train iris";
         this.buttonTrainIris.UseVisualStyleBackColor = true;
         this.buttonTrainIris.Click += new System.EventHandler(this.buttonTrainIris_Click);
         // 
         // labelPrediction
         // 
         this.labelPrediction.AutoSize = true;
         this.labelPrediction.Location = new System.Drawing.Point(12, 39);
         this.labelPrediction.Name = "labelPrediction";
         this.labelPrediction.Size = new System.Drawing.Size(16, 13);
         this.labelPrediction.TabIndex = 3;
         this.labelPrediction.Text = "...";
         // 
         // panelPlot
         // 
         this.panelPlot.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
         this.panelPlot.Location = new System.Drawing.Point(13, 55);
         this.panelPlot.Name = "panelPlot";
         this.panelPlot.Size = new System.Drawing.Size(706, 556);
         this.panelPlot.TabIndex = 4;
         this.panelPlot.Paint += new System.Windows.Forms.PaintEventHandler(this.panelPlot_Paint);
         this.panelPlot.Resize += new System.EventHandler(this.panelPlot_Resize);
         // 
         // buttonPlot
         // 
         this.buttonPlot.Location = new System.Drawing.Point(215, 13);
         this.buttonPlot.Name = "buttonPlot";
         this.buttonPlot.Size = new System.Drawing.Size(75, 23);
         this.buttonPlot.TabIndex = 2;
         this.buttonPlot.Text = "Plot";
         this.buttonPlot.UseVisualStyleBackColor = true;
         this.buttonPlot.Click += new System.EventHandler(this.buttonPlot_Click);
         // 
         // buttonTrainFeet
         // 
         this.buttonTrainFeet.Location = new System.Drawing.Point(94, 13);
         this.buttonTrainFeet.Name = "buttonTrainFeet";
         this.buttonTrainFeet.Size = new System.Drawing.Size(75, 23);
         this.buttonTrainFeet.TabIndex = 1;
         this.buttonTrainFeet.Text = "Train feet";
         this.buttonTrainFeet.UseVisualStyleBackColor = true;
         this.buttonTrainFeet.Click += new System.EventHandler(this.buttonTrainFeet_Click);
         // 
         // MainForm
         // 
         this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
         this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
         this.ClientSize = new System.Drawing.Size(731, 623);
         this.Controls.Add(this.buttonTrainFeet);
         this.Controls.Add(this.buttonPlot);
         this.Controls.Add(this.panelPlot);
         this.Controls.Add(this.labelPrediction);
         this.Controls.Add(this.buttonTrainIris);
         this.Name = "MainForm";
         this.Text = "Form1";
         this.ResumeLayout(false);
         this.PerformLayout();

      }

      #endregion

      private System.Windows.Forms.Button buttonTrainIris;
      private System.Windows.Forms.Label labelPrediction;
      private System.Windows.Forms.Panel panelPlot;
      private System.Windows.Forms.Button buttonPlot;
      private System.Windows.Forms.Button buttonTrainFeet;
   }
}

