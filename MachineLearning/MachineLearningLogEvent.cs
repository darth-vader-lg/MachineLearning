namespace MachineLearning
{
   /// <summary>
   /// Argomenti dell'evento di log
   /// </summary>
   public class MachineLearningLogEventArgs
   {
      #region Properties
      /// <summary>
      /// Tipo di messaggio di log
      /// </summary>
      public MachineLearningLogKind Kind { get; }
      /// <summary>
      /// Sorgente del messaggio
      /// </summary>
      public string Source { get; }
      /// <summary>
      /// Messaggio
      /// </summary>
      public string Message { get; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="message"></param>
      /// <param name="kind"></param>
      /// <param name="source"></param>
      public MachineLearningLogEventArgs(string message, MachineLearningLogKind kind, string source)
      {
         Message = message;
         Kind = kind;
         Source = source;
      }
      #endregion
   }

   /// <summary>
   /// Delegato all'evento di log
   /// </summary>
   /// <param name="sender">Oggetto generante</param>
   /// <param name="e">Argoment dell'evento</param>
   public delegate void MachineLearningLogEventHandler(object sender, MachineLearningLogEventArgs e);

   /// <summary>
   /// Tipo di messaggio di log
   /// </summary>
   public enum MachineLearningLogKind
   {
      Trace = 0,
      Info = 1,
      Warning = 2,
      Error = 3
   }
}
