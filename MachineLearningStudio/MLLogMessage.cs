using Microsoft.ML.Runtime;
using System;

namespace MachineLearningStudio
{
   /// <summary>
   /// Argomenti dell'veto di log
   /// </summary>
   public class MLLogMessageEventArgs : EventArgs
   {
      #region Properties
      public ChannelMessageKind Kind { get; }
      /// <summary>
      /// Messaggio di log
      /// </summary>
      public string Text { get; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="text">Testo di log</param>
      /// <param name="kind">Tipo di messaggio</param>
      public MLLogMessageEventArgs(string text, ChannelMessageKind kind)
      {
         Kind = kind;
         Text = text;
      }
      #endregion
   }

   /// <summary>
   /// Delegato evento di log
   /// </summary>
   /// <param name="sender">Oggetto generatore</param>
   /// <param name="e">Argomenti</param>
   public delegate void MLLogEventHandler(object sender, MLLogMessageEventArgs e);
}
