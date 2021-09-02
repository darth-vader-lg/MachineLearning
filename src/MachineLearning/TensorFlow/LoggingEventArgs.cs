using Microsoft.ML.Runtime;
using System;

namespace MachineLearning.TensorFlow
{
   /// <summary>
   /// Dati dell'evento di log di TensorFlow
   /// </summary>
   public class LoggingEventArgs : EventArgs
   {
      #region Properties
      /// <summary>
      /// La sorgente del messaggio
      /// </summary>
      public string Source { get; }
      /// <summary>
      /// Il tipo di messaggio
      /// </summary>
      public ChannelMessageKind Kind { get; }
      /// <summary>
      /// Il messaggio
      /// </summary>
      public string Message => $"{Kind} {RawMessage}{(!string.IsNullOrEmpty(Source) ? $" {Source}" : "")}"; // TODO: Formattare correttamente
      /// <summary>
      /// Il messaggio che non include la sorgente ed il tipo
      /// </summary>
      public string RawMessage { get; }
      #endregion
      #region Methods
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="message">Messaggio da loggare</param>
      public LoggingEventArgs(string message) => RawMessage = message;
      /// <summary>
      /// Costruttore
      /// </summary>
      /// <param name="message">Messaggio da loggare</param>
      /// <param name="kind">Il tipo di messaggio</param>
      /// <param name="source">La sorgente del messaggio</param>
      public LoggingEventArgs(string message, ChannelMessageKind kind, string source)
      {
         RawMessage = message;
         Kind = kind;
         Source = source;
      }
      #endregion
   }
}
